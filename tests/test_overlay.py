"""An overlay session: the store is read and never written, and rm -rf undoes it."""

import itertools
import json
import os
import shutil
import types
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from runs import run_of
from tracedrepo import fresh_process

import furu
from furu import Added, Spec, provenance, testing
from furu.config import _Config, get_config, layers
from furu.migration.steps import MigrationError
from furu.storage._layout import remembered_dir_in, trace_path_in
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend

_runs = itertools.count()
_calls: list[int] = []
_custom: list[Path] = []  # the storage of Custom, set per test


class Base(Spec):
    n: int

    def create(self) -> int:
        return next(_runs)


class Top(Spec):
    base: Base

    def create(self) -> int:
        return self.base.create() + 1000


class Custom(Spec):
    top: Top

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(storage=_custom[-1])

    def create(self) -> int:
        return self.top.create() + 1000


class TracedBase(Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return next(_runs)


class TracedTop(Spec):
    base: TracedBase

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return self.base.create() + 1000


class Sweep(Spec):
    seeds: int

    @furu.dependency(remember=True)
    def runs(self) -> list[Base]:
        _calls.append(self.seeds)
        return [Base(n=seed) for seed in range(self.seeds)]

    def create(self) -> int:
        return len(furu.load_existing(self.runs))


@pytest.fixture(autouse=True)
def _a_fresh_process(tmp_path: Path) -> Iterator[None]:
    _calls.clear()
    _custom.append(tmp_path / "custom")
    fresh_process()
    yield
    _custom.pop()


def _store() -> list[Path]:
    """Every folder furu writes to outside an overlay, in this test."""
    return [path for _, path in get_config().directories.anchored()] + [_custom[-1]]


def _look(roots: list[Path]) -> dict[str, tuple[bool, int, int]]:
    """Each file and folder under ``roots``: whether a folder, its size and mtime."""
    seen = {}
    for root in roots:
        for folder, dirs, files in os.walk(root):
            for name in [*dirs, *files]:
                path = Path(folder, name)
                stat = path.lstat()
                seen[str(path)] = (path.is_dir(), stat.st_size, stat.st_mtime_ns)
    return seen


def _chmod(roots: list[Path], *, folders: int, files: int) -> None:
    for root in roots:
        if not root.exists():
            continue
        root.chmod(folders)
        for folder, dirs, names in os.walk(root):
            for name in dirs:
                Path(folder, name).chmod(folders)
            for name in names:
                Path(folder, name).chmod(files)


@contextmanager
def _session(overlay: Path) -> Generator[Path]:
    """An overlay session over a store that cannot be written to, and is not."""
    store = _store()
    before = _look(store)
    _chmod(store, folders=0o555, files=0o444)
    fresh_process()
    try:
        with override_config(get_config().model_copy(update={"overlay": overlay})):
            yield overlay
    finally:
        _chmod(store, folders=0o755, files=0o644)
        fresh_process()
    assert _look(store) == before


def test_a_session_builds_on_the_store_and_writes_only_the_overlay(
    tmp_path: Path,
) -> None:
    stored = Base(n=1).create()
    overlay = tmp_path / "overlay"

    with _session(overlay):
        # Every folder furu writes to is in the overlay, at its path in the store.
        for name, path in get_config().directories.anchored():
            assert path == overlay / layers(path)[1].relative_to("/"), name
        assert get_config().manager_version_dir.is_relative_to(overlay)

        assert Base(n=1).done
        top = Custom(top=Top(base=Base(n=1)))
        assert furu.create(top) == stored + 2000
        assert not run_of(Base(n=1)).is_relative_to(overlay)
        assert run_of(Top(base=Base(n=1))).is_relative_to(overlay)
        # Results stored elsewhere are mirrored the same way.
        assert run_of(top).is_relative_to(overlay / _custom[-1].relative_to("/"))
        assert list(get_config().directories.anchored().plans.glob("*.plan"))

        # Workers run in the session too.
        other = Top(base=Base(n=2))
        assert furu.create(other, on=[LocalThreadWorkerBackend()]) >= 1000
        assert run_of(Base(n=2)).is_relative_to(overlay)

    assert (overlay / ".gitignore").read_text().endswith("*\n")
    # The store has no trace of the session.
    assert not Top(base=Base(n=1)).done
    assert not Base(n=2).done

    # Deleting the overlay undoes it.
    shutil.rmtree(overlay)
    with _session(overlay):
        assert Base(n=1).done
        assert not Top(base=Base(n=1)).done


def test_the_overlay_wins_over_the_store(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    with override_config(get_config().model_copy(update={"overlay": overlay})):
        in_overlay = Base(n=7).create()
    in_store = Base(n=7).create()
    assert in_store != in_overlay

    with _session(overlay):
        assert Base(n=7).load_existing() == in_overlay
        assert Top(base=Base(n=7)).create() == in_overlay + 1000
    assert Base(n=7).load_existing() == in_store


def test_a_session_deletes_only_what_it_wrote(tmp_path: Path) -> None:
    Base(n=1).create()
    with _session(tmp_path / "overlay"):
        with pytest.raises(RuntimeError, match="is in the store"):
            Base(n=1).delete("force")
        assert Base(n=1).done

        Base(n=2).create()
        assert Base(n=2).delete("force")
        assert not Base(n=2).done


def test_a_traced_result_built_on_the_store_stays_done(tmp_path: Path) -> None:
    stored = TracedBase(n=1).create()
    with _session(tmp_path / "overlay") as overlay:
        top = TracedTop(base=TracedBase(n=1))
        assert top.create() == stored + 1000
        version = run_of(top).parents[1]
        assert version.is_relative_to(overlay)
        # Named as in the store, so it holds in either layer.
        (built_on,) = json.loads(trace_path_in(version).read_text())["built_on"]
        base_version = run_of(TracedBase(n=1)).parents[1]
        assert not base_version.is_relative_to(overlay)
        assert (
            get_config().directories.anchored().objects / built_on
            == layers(base_version)[0]
        )

        fresh_process()
        assert TracedTop(base=TracedBase(n=1)).done


def test_what_a_dependency_method_returned_is_read_from_the_store(
    tmp_path: Path,
) -> None:
    furu.create(Sweep(seeds=2))
    assert _calls == [2]
    with _session(tmp_path / "overlay") as overlay:
        assert Sweep(seeds=2).done
        assert _calls == [2]

        assert furu.create(Sweep(seeds=3)) == 3
        assert _calls == [2, 3]
        assert remembered_dir_in(Sweep(seeds=3)._class_dir).is_relative_to(overlay)

        fresh_process()
        assert Sweep(seeds=3).done
        assert _calls == [2, 3]


def _video(default: int | None, *, step: bool = True) -> Spec:
    """Video as its source reads: without fps, or with fps added at ``default``."""
    annotations: dict[str, type] = {"clip": str}
    body: dict[str, Any] = {
        "__module__": __name__,
        "__annotations__": annotations,
        "create": lambda self: next(_runs),
    }
    if default is not None:
        annotations["fps"] = int
        body["fps"] = default
        body["migrations"] = (Added("fps", default=default),) if step else ()
    cls = types.new_class("Video", (Spec,), exec_body=lambda ns: ns.update(body))
    return cast(Callable[..., Spec], cls)(clip="x")


def test_a_default_the_store_recorded_still_stops_a_session(tmp_path: Path) -> None:
    _video(30).create()
    with _session(tmp_path / "overlay"):
        assert _video(30).done
        with pytest.raises(MigrationError, match="recorded 30"):
            _video(25).create()


def test_a_session_asks_before_it_leaves_the_stores_results_behind(
    tmp_path: Path,
) -> None:
    _video(None).create()
    # fps without its step: a new class part, which the store's folder is not.
    with (
        _session(tmp_path / "overlay"),
        pytest.raises(MigrationError, match="leave stored results behind"),
    ):
        furu.create(_video(30, step=False))


def test_an_overlay_is_set_by_the_environment_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FURU_OVERLAY", str(tmp_path / "a" / ".." / "overlay"))
    assert _Config().overlay == tmp_path / "overlay"

    monkeypatch.setenv("FURU_OVERLAY", "relative/overlay")
    with pytest.raises(ValidationError, match="absolute"):
        _Config()

    monkeypatch.delenv("FURU_OVERLAY")
    (tmp_path / "pyproject.toml").write_text('[tool.furu]\noverlay = "/tmp/x"\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="FURU_OVERLAY"):
        _Config()


def test_an_overlay_is_a_folder_of_its_own(tmp_path: Path) -> None:
    used = tmp_path / "used"
    used.mkdir()
    (used / "notes.txt").write_text("mine\n")
    with (
        override_config(get_config().model_copy(update={"overlay": used})),
        pytest.raises(RuntimeError, match="did not make"),
    ):
        _ = Base(n=1)._class_dir
    assert sorted(os.listdir(used)) == ["notes.txt"]

    empty = tmp_path / "empty"
    empty.mkdir()
    with override_config(get_config().model_copy(update={"overlay": empty})):
        assert Base(n=1)._class_dir.is_relative_to(empty)
    assert os.listdir(empty) == [".gitignore"]


def test_tests_never_run_in_the_shells_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FURU_OVERLAY", str(tmp_path / "overlay"))
    monkeypatch.setattr(provenance, "_require_pushed_code", False)
    plugin = cast(pytest.Config, types.SimpleNamespace(stash=pytest.Stash()))
    with override_config(_Config()):
        assert get_config().overlay == tmp_path / "overlay"
        testing.pytest_configure(plugin)
        try:
            assert "FURU_OVERLAY" not in os.environ
            assert get_config().overlay is None
        finally:
            testing.pytest_unconfigure(plugin)
