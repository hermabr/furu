"""Overlay / debug sessions: read the base store read-only, write only to the overlay."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock

import pytest
from runs import run_of, working_folder
from tracedrepo import Repo, fresh_process
from websockets.sync.server import ServerConnection

import furu
from furu import Added, Metadata, Spec, plan
from furu.config import _Config, _FuruDirectories, _project_anchor, get_config
from furu.execution.manager import Manager, _Script
from furu.execution.run import Run
from furu.migration.steps import MigrationError
from furu.provenance import GitIdentity
from furu.snapshot import CodeLocation, require_pushed
from furu.storage._layout import (
    remembered_dir_in,
    schema_path_in,
    trace_path_in,
)
from furu.storage.attempt_lock import HeldError
from furu.testing import _replace_config_directories, override_config
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.messages import JobCompletedResult, Launch

_created: list[str] = []
_remember_calls: list[int] = []


class Corpus(Spec):
    name: str

    def create(self) -> str:
        _created.append(self.name)
        return self.name.upper()


class Tokens(Spec):
    corpus: Corpus

    def create(self) -> Path:
        path = self.directory.data / "tokens.txt"
        path.write_text(self.corpus.create())
        return path


class Kept(Spec):
    root: str
    n: int

    def metadata(self) -> Metadata:
        return Metadata(storage=Path(self.root))

    def create(self) -> int:
        return self.n


class Stops(Spec):
    def create(self) -> int:
        if get_config().overlay is None:
            raise KeyboardInterrupt
        return 1


class Pair(Spec):
    n: int

    @furu.batched(lambda self, worker: ("pairs", 2))
    def create(specs: list[Pair], directory: furu.SpecDirectory) -> list[int]:
        return [spec.n for spec in specs]


class Versioned(Spec):
    n: int
    value: ClassVar[str] = "base"

    def create(self) -> str:
        return f"{Versioned.value}:{self.n}"


class BatchedVersioned(Spec):
    n: int
    value: ClassVar[str] = "base"

    @furu.batched(lambda self, worker: ("bv", 4))
    def create(
        specs: list[BatchedVersioned], directory: furu.SpecDirectory
    ) -> list[str]:
        return [f"{BatchedVersioned.value}:{spec.n}" for spec in specs]


class Remembers(Spec):
    n: int
    returned: ClassVar[str] = "base"

    @furu.dependency(remember=True)
    def parts(self) -> list[Corpus]:
        _remember_calls.append(self.n)
        return [Corpus(name=f"{Remembers.returned}-{self.n}")]

    def create(self) -> str:
        return self.parts[0].create()


@pytest.fixture(autouse=True)
def _reset_globals() -> None:
    _created.clear()
    _remember_calls.clear()
    Versioned.value = "base"
    BatchedVersioned.value = "base"
    Remembers.returned = "base"


def _overlaid(path: Path) -> _Config:
    return get_config().model_copy(update={"overlay": path})


def _snapshot_dirs(*roots: Path) -> dict[Path, bytes | None]:
    return {
        path: path.read_bytes() if path.is_file() else None
        for root in roots
        if root.exists()
        for path in sorted(root.rglob("*"))
    }


def _base_dirs() -> tuple[Path, ...]:
    return tuple(path for _, path in get_config().directories.anchored())


@contextmanager
def _readonly_tree(*roots: Path) -> Generator[None]:
    """OS-level read-only permissions on existing base trees (files 0o444, dirs 0o555)."""
    saved: list[tuple[Path, int]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in [root, *root.rglob("*")]:
            saved.append((path, stat.S_IMODE(path.stat().st_mode)))
    # Lock files first, then directories bottom-up so traversal still works.
    for path, mode in reversed(saved):
        if path.is_file():
            os.chmod(path, mode & ~0o222)
    for path, mode in reversed(saved):
        if path.is_dir():
            os.chmod(path, mode & ~0o222)
    try:
        yield
    finally:
        # Restore directories top-down first, then files.
        for path, mode in saved:
            if path.is_dir() and path.exists():
                os.chmod(path, mode)
        for path, mode in saved:
            if path.is_file() and path.exists():
                os.chmod(path, mode)


def _mirror(overlay: Path, path: Path) -> Path:
    return overlay / path.relative_to(path.anchor)


# --- Core read-through, write isolation, and OS read-only base ----------------


def test_overlay_loads_base_read_only_and_writes_only_to_itself(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    assert Corpus(name="a").create() == "A"
    before = _snapshot_dirs(*_base_dirs())
    dirs = get_config().directories.anchored()

    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        corpus, tokens = Corpus(name="a"), Tokens(corpus=Corpus(name="a"))
        path = tokens.create()

        assert path.read_text() == "A"
        assert _created == ["a"]  # loaded from read-only base
        assert run_of(corpus).is_relative_to(dirs.objects)
        assert run_of(tokens).is_relative_to(_mirror(overlay, dirs.objects))
        assert path.is_relative_to(run_of(tokens))
        assert corpus.done and tokens.done

    assert _snapshot_dirs(*_base_dirs()) == before
    assert not Tokens(corpus=Corpus(name="a")).done
    for written in (dirs.plans, dirs.provenance):
        assert list(_mirror(overlay, written).iterdir())

    shutil.rmtree(overlay)
    with override_config(_overlaid(overlay)):
        assert Corpus(name="a").done
        assert not Tokens(corpus=Corpus(name="a")).done


def test_custom_storage_is_mirrored_and_base_stays_read_only(tmp_path: Path) -> None:
    root = tmp_path / "kept"
    overlay = tmp_path / "overlay"
    assert Kept(root=str(root), n=1).create() == 1
    before = _snapshot_dirs(root)

    with _readonly_tree(root, *_base_dirs()), override_config(_overlaid(overlay)):
        assert run_of(Kept(root=str(root), n=1)).is_relative_to(root)
        new = Kept(root=str(root), n=2)
        assert new.create() == 2
        assert run_of(new).is_relative_to(_mirror(overlay, root))

    assert _snapshot_dirs(root) == before


def test_workers_inherit_overlay_and_never_touch_base(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    Corpus(name="a").create()
    before = _snapshot_dirs(*_base_dirs())
    dirs = get_config().directories.anchored()

    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        tokens = Tokens(corpus=Corpus(name="a"))
        path = furu.create(tokens, on=[LocalThreadWorkerBackend()])

        assert path.read_text() == "A"
        assert path.is_relative_to(overlay)
        assert list(_mirror(overlay, dirs.executions).glob("*/execution.log"))

    assert _snapshot_dirs(*_base_dirs()) == before


def test_held_working_folder_in_base_does_not_block_overlay(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        Stops().create()
    folder = working_folder(Stops())
    folder.rename(folder.with_name("local__elsewhere__1__token"))
    with pytest.raises(HeldError, match="process 1 on elsewhere"):
        Stops().create()
    before = _snapshot_dirs(*_base_dirs())

    with _readonly_tree(*_base_dirs()), override_config(_overlaid(tmp_path / "overlay")):
        assert Stops().create() == 1

    assert _snapshot_dirs(*_base_dirs()) == before


# --- Layer precedence: overlay always wins over base --------------------------


def test_overlay_takes_precedence_over_base_for_unbatched_specs(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    # Create an older result in base.
    Versioned.value = "base"
    assert Versioned(n=1).create() == "base:1"

    # Build the same spec into overlay while base is absent, then restore base.
    empty_base = tmp_path / "empty-base"
    real_cfg = get_config()
    Versioned.value = "overlay"
    with override_config(
        real_cfg.model_copy(
            update={
                "directories": _FuruDirectories(
                    **{name: empty_base / name for name in _FuruDirectories.model_fields}
                ),
            }
        )
    ):
        assert Versioned(n=1).create() == "overlay:1"
        src = Versioned(n=1)._class_dir
    dst = _mirror(overlay, Versioned(n=1)._class_dir)
    shutil.copytree(src, dst)

    # Now BOTH base (older) and overlay (newer) hold Versioned(n=1).
    # Overlay MUST win instead of pooling both layers and picking the older base run.
    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        assert Versioned(n=1).load_existing() == "overlay:1"
        assert run_of(Versioned(n=1)).is_relative_to(overlay)

    assert Versioned(n=1).load_existing() == "base:1"


def test_overlay_takes_precedence_over_base_for_batched_specs(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    BatchedVersioned.value = "base"
    assert furu.create([BatchedVersioned(n=1), BatchedVersioned(n=2)]) == [
        "base:1",
        "base:2",
    ]
    before = _snapshot_dirs(*_base_dirs())

    # Normal overlay batch: loads base specs, creates only the new spec in overlay.
    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        assert furu.create([Pair(n=1), Pair(n=2)]) == [1, 2]

    # Put a newer batch run for n=1 into overlay so BOTH layers hold n=1.
    empty_base = tmp_path / "empty-base"
    BatchedVersioned.value = "overlay"
    with override_config(
        get_config().model_copy(
            update={
                "directories": _FuruDirectories(
                    **{name: empty_base / name for name in _FuruDirectories.model_fields}
                ),
            }
        )
    ):
        assert furu.create([BatchedVersioned(n=1)]) == ["overlay:1"]
        src = BatchedVersioned(n=1)._class_dir
    shutil.copytree(src, _mirror(overlay, BatchedVersioned(n=1)._class_dir))

    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        assert BatchedVersioned(n=1).load_existing() == "overlay:1"
        assert run_of(BatchedVersioned(n=1)).is_relative_to(overlay)
        assert BatchedVersioned(n=2).load_existing() == "base:2"
        assert not run_of(BatchedVersioned(n=2)).is_relative_to(overlay)

    assert _snapshot_dirs(*_base_dirs()) == before


def test_remember_reads_base_and_prefers_overlay_over_newer_base_mtime(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    Remembers.returned = "base"
    assert Remembers(n=1).create() == "BASE-1"
    assert _remember_calls == [1]
    base_rem_dir = remembered_dir_in(Remembers(n=1)._class_dir)
    before = _snapshot_dirs(*_base_dirs())

    fresh_process()
    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        # Base remembered line holds without calling the method or writing to base.
        assert Remembers(n=1).done
        assert _remember_calls == [1]

        # A new spec runs and saves its remembered line only in overlay.
        assert Remembers(n=2).create() == "BASE-2"
        assert _remember_calls == [1, 2]
        assert remembered_dir_in(Remembers(n=2)._class_dir).is_relative_to(overlay)

    assert _snapshot_dirs(*_base_dirs()) == before

    # Create a conflicting live remembered line for n=1 in overlay with an OLDER
    # mtime than base's file, and verify overlay STILL takes precedence.
    ov_rem_dir = remembered_dir_in(_mirror(overlay, Remembers(n=1)._class_dir))
    (ov_rem_dir / "traces").mkdir(parents=True, exist_ok=True)
    base_launch_file = next(base_rem_dir.glob("launch-*.jsonl"))
    base_line = json.loads(base_launch_file.read_text(encoding="utf-8").splitlines()[0])
    # Copy trace file to overlay and write a line with a different returned fingerprint.
    trace_name = base_line["trace"]
    shutil.copy2(
        base_rem_dir / "traces" / f"{trace_name}.json",
        ov_rem_dir / "traces" / f"{trace_name}.json",
    )
    ov_line = {**base_line, "returned": "overlay-fingerprint"}
    ov_launch_file = ov_rem_dir / "launch-older.jsonl"
    ov_launch_file.write_text(json.dumps(ov_line) + "\n", encoding="utf-8")
    # Make overlay file strictly OLDER than base file.
    base_mtime = base_launch_file.stat().st_mtime
    os.utime(ov_launch_file, (base_mtime - 100, base_mtime - 100))

    fresh_process()
    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        spec = Remembers(n=1)
        assert spec._returned["parts"] == "overlay-fingerprint"


# --- Schema checks: read-only base, no headerless .done write, both checked ---


def _versioned_spec(name: str, default: int | None) -> Callable[..., Spec]:
    annotations: dict[str, Any] = {"clip": str}
    ns: dict[str, Any] = {
        "__module__": __name__,
        "__annotations__": annotations,
        "create": lambda self: getattr(self, "fps", 0),
    }
    if default is not None:
        annotations["fps"] = int
        ns["fps"] = default
        ns["migrations"] = (Added("fps", default=default),)
    return cast(
        Callable[..., Spec],
        types.new_class(name, (Spec,), exec_body=lambda d: d.update(ns)),
    )


def test_schema_base_is_readonly_done_does_not_write_and_both_layers_checked(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    v0 = _versioned_spec("Clip", None)
    assert v0(clip="a").create() == 0
    base_class_dir = v0(clip="a")._class_dir
    base_schema = schema_path_in(base_class_dir)
    before = _snapshot_dirs(*_base_dirs())

    # 1) Reading .done with a newly declared Added step must NOT write a headerless
    #    schema.ndjson in the overlay (or touch the read-only base).
    v30 = _versioned_spec("Clip", 30)
    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        assert v30(clip="a").done
        ov_schema = schema_path_in(v30(clip="a")._class_dir)
        assert not ov_schema.exists()

    # 2) Creating a new spec in the overlay writes a valid schema.ndjson WITH its
    #    {"names": ...} header and the Added step, leaving base untouched.
    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        assert v30(clip="b").create() == 30
        ov_schema = schema_path_in(v30(clip="b")._class_dir)
        lines = [json.loads(line) for line in ov_schema.read_text().splitlines()]
        assert lines == [
            {"names": ["clip"]},
            {"added": "fps", "default": 30},
        ]
    assert _snapshot_dirs(*_base_dirs()) == before

    # 3) Now base records a DIFFERENT default (25) for the same step.
    #    In the overlay, NEITHER conflict may be masked, and each error must point
    #    to the actual conflicting file!
    v25 = _versioned_spec("Clip", 25)
    assert v25(clip="c").create() == 25

    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        # Declaring 30 conflicts with base (recorded 25) -> error names base_schema!
        with pytest.raises(MigrationError) as exc_base:
            _ = _versioned_spec("Clip", 30)(clip="a").done
        assert str(base_schema) in str(exc_base.value)
        assert "recorded 25" in str(exc_base.value)

        # Declaring 25 matches base, but conflicts with overlay (recorded 30) ->
        # error names ov_schema!
        with pytest.raises(MigrationError) as exc_ov:
            _ = _versioned_spec("Clip", 25)(clip="a").done
        assert str(ov_schema) in str(exc_ov.value)
        assert "recorded 30" in str(exc_ov.value)


# --- launch.ask() across both layers -----------------------------------------


@dataclass(frozen=True)
class _NestedV1:
    a: int


@dataclass(frozen=True)
class _NestedV2:
    a: int
    b: int = 0


def test_launch_ask_checks_unexplained_class_and_nested_schemas_in_base(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"

    # Unexplained class folder in base.
    before_cls = cast(
        Callable[..., Spec],
        types.new_class(
            "AskSpec",
            (Spec,),
            exec_body=lambda ns: ns.update(
                __module__=__name__,
                __annotations__={"x": int},
                create=lambda self: self.x,
            ),
        ),
    )
    after_cls = cast(
        Callable[..., Spec],
        types.new_class(
            "AskSpec",
            (Spec,),
            exec_body=lambda ns: ns.update(
                __module__=__name__,
                __annotations__={"x": int, "y": int},
                y=1,
                create=lambda self: self.x + self.y,
            ),
        ),
    )
    assert before_cls(x=1).create() == 1
    base_dir = before_cls(x=1)._class_dir

    with (
        _readonly_tree(*_base_dirs()),
        override_config(_overlaid(overlay)),
        pytest.raises(MigrationError, match="would leave stored results behind") as exc,
    ):
        after_cls(x=1).create()
    assert str(base_dir) in str(exc.value)

    # Unexplained nested dataclass schema recorded in base.
    nested_v1 = cast(
        Callable[..., Spec],
        types.new_class(
            "HoldsNested",
            (Spec,),
            exec_body=lambda ns: ns.update(
                __module__=__name__,
                __annotations__={"cfg": _NestedV1},
                create=lambda self: self.cfg.a,
            ),
        ),
    )
    _NestedV2.__qualname__ = _NestedV1.__qualname__
    _NestedV2.__name__ = _NestedV1.__name__
    nested_v2 = cast(
        Callable[..., Spec],
        types.new_class(
            "HoldsNested",
            (Spec,),
            exec_body=lambda ns: ns.update(
                __module__=__name__,
                __annotations__={"cfg": _NestedV2},
                create=lambda self: self.cfg.a + self.cfg.b,
            ),
        ),
    )
    assert nested_v1(cfg=_NestedV1(a=1)).create() == 1
    nested_base_dir = nested_v1(cfg=_NestedV1(a=1))._class_dir

    with (
        _readonly_tree(*_base_dirs()),
        override_config(_overlaid(overlay)),
        pytest.raises(MigrationError, match="would leave stored results behind") as exc2,
    ):
        nested_v2(cfg=_NestedV2(a=1, b=2)).create()
    assert str(nested_base_dir) in str(exc2.value)


# --- Spec.delete() in overlay: block live base runs, allow outdated ones ------


CHILD_SRC = """\
import furu


class Child(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return self.n + 1
"""

PARENT_SRC = """\
import furu

from PKG.child import Child


class Parent(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return Child(n=self.n).create() * 10
"""


def test_delete_blocks_live_base_run_but_allows_deleting_over_outdated_base_run(
    repo: Repo, tmp_path: Path
) -> None:
    overlay = tmp_path / "overlay"
    # Fixed spec: live in base -> blocked; overlay-only -> deleted.
    Corpus(name="a").create()
    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        with pytest.raises(RuntimeError, match="live result"):
            Corpus(name="a").delete("force")
        new = Corpus(name="b")
        new.create()
        assert new.delete("force")
        assert not new.done
    assert Corpus(name="a").done

    # Traced spec: live in base -> blocked; outdated in base -> allowed!
    repo.write("child.py", CHILD_SRC)
    child_mod = repo.load("child")
    assert child_mod.Child(n=1).create() == 2

    with (
        _readonly_tree(*_base_dirs()),
        override_config(_overlaid(overlay)),
        pytest.raises(RuntimeError, match="live result"),
    ):
        child_mod.Child(n=1).delete("force")

    # Edit child.py so the base run is now OUTDATED.
    repo.edit("child.py", "self.n + 1", "self.n + 2")
    child_mod = repo.load("child")
    before = _snapshot_dirs(*_base_dirs())

    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        spec = child_mod.Child(n=1)
        assert not spec.done
        assert spec.create() == 3
        assert run_of(spec).is_relative_to(overlay)
        assert spec.delete("force")
        assert not spec.done

    assert _snapshot_dirs(*_base_dirs()) == before


# --- Code tracing: canonical built_on & overlay in _Collector.data_dirs -------


def test_traced_canonical_built_on_across_layers_and_custom_storage(
    repo: Repo, tmp_path: Path
) -> None:
    overlay = tmp_path / "overlay"
    repo.write("child.py", CHILD_SRC)
    repo.write("parent.py", PARENT_SRC)
    parent_mod = repo.load("parent")
    child = parent_mod.Child(n=1)
    assert child.create() == 2
    child_rel = os.path.relpath(run_of(child).parent.parent, child._metadata.storage)
    before = _snapshot_dirs(*_base_dirs())

    with _readonly_tree(*_base_dirs()), override_config(_overlaid(overlay)):
        parent = parent_mod.Parent(n=1)
        assert parent.create() == 20
        assert run_of(parent).is_relative_to(overlay)
        trace = json.loads(trace_path_in(run_of(parent).parent.parent).read_text())
        assert trace["built_on"] == [child_rel]
        fresh_process()
        assert parent_mod.Parent(n=1).done

        # Editing upstream child invalidates the overlay parent built on base child.
        repo.edit("child.py", "self.n + 1", "self.n + 2")
        parent_mod = repo.load("parent")
        assert not parent_mod.Parent(n=1).done
        assert parent_mod.Parent(n=1).create() == 30
        assert run_of(parent_mod.Child(n=1)).is_relative_to(overlay)

    assert _snapshot_dirs(*_base_dirs()) == before

    # Also verify cross-storage canonical built_on matches non-overlay.
    other_storage = tmp_path / "other-storage"
    repo.write(
        "cross.py",
        f"""\
from pathlib import Path
import furu


class Elsewhere(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(storage=Path({str(other_storage)!r}), code_version="traced")

    def create(self) -> int:
        return self.n + 1


class Here(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return Elsewhere(n=self.n).create() + 100
""",
    )
    cross = repo.load("cross")
    assert cross.Elsewhere(n=1).create() == 2

    with _readonly_tree(other_storage, *_base_dirs()), override_config(_overlaid(overlay)):
        here_ov = cross.Here(n=1)
        assert here_ov.create() == 102
        ov_version = run_of(here_ov).parent.parent.name
        ov_trace = json.loads(trace_path_in(run_of(here_ov).parent.parent).read_text())
        fresh_process()
        assert cross.Here(n=1).done

    here_base = cross.Here(n=1)
    assert not here_base.done
    assert here_base.create() == 102
    base_version = run_of(here_base).parent.parent.name
    base_trace = json.loads(trace_path_in(run_of(here_base).parent.parent).read_text())
    assert (ov_version, ov_trace["built_on"]) == (base_version, base_trace["built_on"])
    assert ov_trace["built_on"][0].startswith("..")


def test_collector_skips_reads_inside_in_repo_overlay(
    repo: Repo, caplog: pytest.LogCaptureFixture
) -> None:
    overlay = repo.root / "debug-overlay"
    repo.write(
        "reader.py",
        """\
from pathlib import Path
import furu


class Writer(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> Path:
        p = self.directory.data / "out.txt"
        p.write_text("hello")
        return p


class Reader(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> str:
        return Writer().create().read_text()
""",
    )
    mod = repo.load("reader")
    with caplog.at_level("WARNING"), override_config(_overlaid(overlay)):
        assert mod.Reader().create() == "hello"
    assert not any("does not trace" in rec.message for rec in caplog.records)


# --- Shared Manager isolation across real / multiple overlays -----------------


def test_shared_manager_isolates_real_and_overlay_launches(tmp_path: Path) -> None:
    backend = LocalThreadWorkerBackend(max_workers=1)
    pool_key = backend.pool_key
    manager = Manager(url="ws://unused", manager_dir=None)
    leaf = Corpus(name="shared")
    base_cfg = get_config()
    ov1 = tmp_path / "ov1"
    ov2 = tmp_path / "ov2"

    def submit_with(cfg: _Config, name: str) -> tuple[_Script, Run]:
        with override_config(cfg):
            obj = Corpus(name="shared")
            run = Run(
                backends={pool_key: backend},
                launch=Launch(
                    plan=tmp_path / f"{name}.plan",
                    execution_log=tmp_path / f"{name}.log",
                    config=cfg.anchored(),
                    code=CodeLocation(
                        python=Path(sys.executable),
                        project_root=tmp_path,
                        cwd=tmp_path,
                        repo_root=tmp_path,
                        checkout=tmp_path,
                    ),
                ),
            )
            groups = run.place([obj])
            lines = list(map(plan.dump, plan.lines([obj], {obj.object_id})))
            submit = run.submit(groups, lines, tmp_path / f"{name}.plan")
        script = _Script(cast(ServerConnection, MagicMock()))
        manager.scripts.append(script)
        manager._add(script, submit)
        return script, run

    # Real + ov1 (two scripts) + ov2: 3 distinct jobs; the two ov1 scripts share 1.
    s_real, _ = submit_with(base_cfg, "real")
    s_ov1_a, _ = submit_with(_overlaid(ov1), "ov1-a")
    s_ov1_b, _ = submit_with(_overlaid(ov1), "ov1-b")
    s_ov2, _ = submit_with(_overlaid(ov2), "ov2")

    leases = []
    while (
        lease := manager._hand_out(cast(ServerConnection, object()), pool_key)
    ) is not None:
        leases.append(lease)

    assert len(leases) == 3
    assert [lease.job.config.overlay for lease in leases] == [None, ov1, ov2]
    assert [lease.job.specs for lease in leases] == [
        [leaf.object_id],
        [leaf.object_id],
        [leaf.object_id],
    ]
    assert leases[0].job.class_dir == leaf._class_dir
    assert leases[1].job.class_dir.is_relative_to(ov1)
    assert leases[2].job.class_dir.is_relative_to(ov2)

    # Finishing ov1 notifies BOTH ov1 scripts and neither real nor ov2.
    worker_ov1 = next(w for w, held in manager.leases.items() if held is leases[1])
    del manager.leases[worker_ov1]
    manager._finish(leases[1], JobCompletedResult())
    assert [script for script, _ in manager.outbox] == [s_ov1_a, s_ov1_b]
    assert ov1 not in manager.scopes  # pruned when empty

    # Dropping/cancelling ov2 leaves real unaffected.
    manager.outbox.clear()
    worker_ov2 = next(w for w, held in manager.leases.items() if held is leases[2])
    del manager.leases[worker_ov2]
    manager._finish(leases[2], JobCompletedResult())
    assert [script for script, _ in manager.outbox] == [s_ov2]
    assert ov2 not in manager.scopes

    manager.outbox.clear()
    worker_real = next(w for w, held in manager.leases.items() if held is leases[0])
    del manager.leases[worker_real]
    manager._finish(leases[0], JobCompletedResult())
    assert [script for script, _ in manager.outbox] == [s_real]
    assert not manager.scopes


# --- Path guards, normalization, empty env var, git check, testing hygiene ----


def test_overlay_env_empty_normalization_and_anchoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FURU_OVERLAY", "")
    assert _Config().overlay is None

    monkeypatch.setenv("FURU_OVERLAY", "   ")
    assert _Config().overlay is None

    monkeypatch.setenv("FURU_OVERLAY", "debug-dir/")
    cfg = _Config()
    assert cfg.overlay == Path("debug-dir")
    assert cfg.anchored().overlay == _project_anchor() / "debug-dir"

    raw = f"{tmp_path}/a/../b//"
    monkeypatch.setenv("FURU_OVERLAY", raw)
    cfg_abs = _Config()
    assert cfg_abs.overlay == tmp_path / "b"
    real = Path("/data/objects/Leaf")
    assert cfg_abs.overlaid(real) == tmp_path / "b" / "data" / "objects" / "Leaf"
    assert cfg_abs.layers(cfg_abs.overlaid(real)) == (cfg_abs.overlaid(real), real)
    assert cfg_abs.layers(real) == (real,)


def test_overlay_rejects_root_repo_root_and_overlapping_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Filesystem root and variants.
    for bad in ("/", "//", "/tmp/.."):
        monkeypatch.setenv("FURU_OVERLAY", bad)
        with pytest.raises(ValueError, match="root"):
            _Config()

    # Relative repo root.
    for bad in (".", "./", "sub/.."):
        monkeypatch.setenv("FURU_OVERLAY", bad)
        with pytest.raises(ValueError, match="root"):
            _Config()

    # Absolute repo root and ancestor of repo root.
    anchor = _project_anchor()
    for bad_path in (anchor, anchor.parent):
        monkeypatch.setenv("FURU_OVERLAY", str(bad_path))
        with pytest.raises(ValueError, match="repository root"):
            _Config()

    # Relative overlap with default furu-data directories.
    for bad in ("furu-data", "furu-data/objects", "furu-data/objects/sub"):
        monkeypatch.setenv("FURU_OVERLAY", bad)
        with pytest.raises(ValueError, match="overlap"):
            _Config()

    monkeypatch.delenv("FURU_OVERLAY", raising=False)

    # Absolute overlap with configured data directories (both directions + equal).
    base_objects = get_config().directories.anchored().objects
    for bad_path in (base_objects, base_objects.parent, base_objects / "sub"):
        with pytest.raises(ValueError, match="overlap"):
            _overlaid(bad_path).anchored()

    # Overlap with custom Metadata(storage=...).
    custom = tmp_path / "custom"
    for bad_path in (custom, custom.parent, custom / "sub"):
        with override_config(_overlaid(bad_path)), pytest.raises(ValueError, match="overlap"):
            _ = Kept(root=str(custom), n=1)._class_dir


def test_require_pushed_gitignore_and_repo_root_guard(git_repo: Path) -> None:
    data = get_config().model_dump()
    data["directories"] = {
        name: git_repo / "furu-data" / name for name in data["directories"]
    }
    data["overlay"] = git_repo / "debug"
    config = _Config.model_validate(data)
    result = config.write_directories.objects / "result.txt"
    result.parent.mkdir(parents=True)
    result.write_text("x\n")

    with override_config(config):
        require_pushed(GitIdentity.capture(git_repo))
    assert not (git_repo / "furu-data").exists()
    assert (git_repo / "debug" / ".gitignore").read_text() == "*\n"

    # Repo-root overlay is refused by require_pushed even when data dirs are elsewhere.
    bad_cfg = get_config().model_copy(update={"overlay": git_repo})
    with override_config(bad_cfg), pytest.raises(ValueError, match="repository root"):
        require_pushed(GitIdentity.capture(git_repo))


def test_testing_clears_overlay(tmp_path: Path) -> None:
    cfg = _overlaid(tmp_path / "overlay")
    assert cfg.overlay == tmp_path / "overlay"
    cleaned = _replace_config_directories(cfg, cfg.directories)
    assert cleaned.overlay is None


def test_furu_overlay_env_var_subprocess_end_to_end(tmp_path: Path) -> None:
    assert Corpus(name="env").create() == "ENV"
    base = get_config().directories.anchored()
    ov = tmp_path / "ov"
    before = _snapshot_dirs(*_base_dirs())

    code = """\
from furu import provenance
from test_overlay import Corpus, Tokens

provenance._require_pushed_code = False
c = Corpus(name="env")
assert c.done
assert c.load_existing() == "ENV"
t = Tokens(corpus=c)
p = t.create()
assert p.read_text() == "ENV"
print("OK")
"""
    env = os.environ | {
        "FURU_OVERLAY": str(ov),
        "FURU_DIRECTORIES__OBJECTS": str(base.objects),
        "FURU_DIRECTORIES__EXECUTIONS": str(base.executions),
        "FURU_DIRECTORIES__PLANS": str(base.plans),
        "FURU_DIRECTORIES__PROVENANCE": str(base.provenance),
        "FURU_DIRECTORIES__SNAPSHOTS": str(base.snapshots),
    }
    with _readonly_tree(*_base_dirs()):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    assert result.stdout.strip().endswith("OK")
    assert _snapshot_dirs(*_base_dirs()) == before
    assert not Tokens(corpus=Corpus(name="env")).done
    assert _mirror(ov, base.objects).is_dir()
