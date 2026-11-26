"""Read-only base + writable overlay mode (``furu.overlay`` and ``FURU_OVERLAY``)."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from gitrepo import commit_and_push, git
from tracedrepo import Repo

import furu
from furu import Metadata, Spec, plan, provenance
from furu.config import _Config, _FuruDirectories, _project_anchor, get_config
from furu.execution.manager import Manager
from furu.execution.run import Run
from furu.migration.steps import MigrationError
from furu.snapshot import CodeLocation, extract_snapshot
from furu.storage._layout import (
    FIXED,
    remembered_dir_in,
    schema_path_in,
)
from furu.storage.stored import stored_row
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.messages import JobCompletedResult, Launch

_leaf_calls: list[str] = []
_parent_calls: list[str] = []
_batch_calls: list[list[int]] = []
_remember_calls: list[int] = []


class Leaf(Spec):
    name: str

    def create(self) -> str:
        _leaf_calls.append(self.name)
        return f"leaf:{self.name}"


class Parent(Spec):
    leaf: Leaf
    tag: str

    def create(self) -> str:
        _parent_calls.append(self.tag)
        return f"{self.leaf.create()}+{self.tag}"


class DynamicParent(Spec):
    name: str
    tag: str

    def create(self) -> str:
        leaf = Leaf(name=self.name)
        return f"dyn:{leaf.create()}+{self.tag}"


class BatchedItem(Spec):
    value: int

    @furu.batched(lambda self, worker: ("items", 4))
    def create(specs: list[BatchedItem], directory: furu.SpecDirectory) -> list[int]:
        _batch_calls.append([s.value for s in specs])
        return [s.value * 10 for s in specs]


class DataLeaf(Spec):
    name: str

    def create(self) -> Path:
        out = self.directory.data / f"{self.name}.txt"
        out.write_text(f"payload:{self.name}", encoding="utf-8")
        return out


class DataParent(Spec):
    leaf: DataLeaf
    suffix: str

    def create(self) -> Path:
        src = self.leaf.create()
        out = self.directory.data / f"out-{self.suffix}.txt"
        out.write_text(f"{src.read_text(encoding='utf-8')}:{self.suffix}", encoding="utf-8")
        return out


class CustomLeaf(Spec):
    name: str
    root: Path

    def metadata(self) -> Metadata:
        return Metadata(storage=self.root)

    def create(self) -> str:
        _leaf_calls.append(f"custom:{self.name}")
        return f"custom:{self.name}"


class RememberSweep(Spec):
    count: int

    @furu.dependency(remember=True)
    def leaves(self) -> list[Leaf]:
        _remember_calls.append(self.count)
        return [Leaf(name=f"r{i}") for i in range(self.count)]

    def create(self) -> list[str]:
        return furu.load_existing(self.leaves)


@contextmanager
def _readonly_tree(*roots: Path):
    """Make existing directories and files under ``roots`` 0o555 / 0o444 so any
    write, lock, mkdir, or unlink fails with PermissionError at the OS level."""
    changed: list[tuple[Path, int]] = []
    for root in roots:
        if not root.exists():
            continue
        entries = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        for path in (*entries, root):
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            changed.append((path, mode))
            path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        yield
    finally:
        for path, mode in reversed(changed):
            if path.exists():
                path.chmod(mode)


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _all_base_dirs() -> list[Path]:
    return [path for _, path in get_config().directories.anchored()]


@pytest.fixture(autouse=True)
def _reset_counters() -> None:
    _leaf_calls.clear()
    _parent_calls.clear()
    _batch_calls.clear()
    _remember_calls.clear()


# --- Core read-only base + writable overlay ----------------------------------


def test_overlay_reads_readonly_base_and_isolates_all_writes(tmp_path: Path) -> None:
    leaf = Leaf(name="base")
    assert leaf.create() == "leaf:base"
    assert _leaf_calls == ["base"]

    base_dirs = _all_base_dirs()
    before = {str(d): _snapshot_tree(d) for d in base_dirs}
    ov = tmp_path / "ov"

    # Instantiated and checked BEFORE entering overlay to ensure no stale caching.
    parent = Parent(leaf=leaf, tag="exp1")
    assert not parent.done

    with _readonly_tree(*base_dirs), furu.overlay(ov) as active_ov:
        assert active_ov == ov
        assert leaf.done
        assert leaf.load_existing() == "leaf:base"
        assert not parent.done

        assert parent.create() == "leaf:base+exp1"
        assert _leaf_calls == ["base"]  # cache hit from base
        assert _parent_calls == ["exp1"]
        assert parent.done

        # Second call hits overlay cache.
        assert parent.create() == "leaf:base+exp1"
        assert _parent_calls == ["exp1"]

        # Provenance & plans landed in overlay, with overlay recorded.
        prov_files = list((ov / "provenance").iterdir())
        assert prov_files
        launch_files = [p for p in prov_files if p.name.startswith("launch-")]
        assert len(launch_files) == 1
        header = plan.read(launch_files[0])[0]
        assert header["overlay"] == str(ov)
        worker_files = [p for p in prov_files if not p.name.startswith("launch-")]
        run_lines = [
            line for wf in worker_files for line in plan.read(wf) if "run" in line
        ]
        assert len(run_lines) == 1
        assert Path(run_lines[0]["storage"]) == ov / "objects"
        assert list((ov / "plans").glob("*.plan"))

    # Base is byte-for-byte identical.
    assert {str(d): _snapshot_tree(d) for d in base_dirs} == before

    # Outside overlay: base result still exists, overlay result does not.
    assert leaf.done
    assert not parent.done

    # Re-entering the same overlay reuses what was written there.
    with furu.overlay(ov):
        assert parent.done
        assert parent.load_existing() == "leaf:base+exp1"
        assert _parent_calls == ["exp1"]

    # Deleting the overlay cleanly erases the session.
    shutil.rmtree(ov)
    with furu.overlay(ov):
        assert leaf.done
        assert not parent.done


def test_overlay_without_path_uses_temp_dir_and_cleans_up(tmp_path: Path) -> None:
    leaf = Leaf(name="base-tmp")
    leaf.create()
    parent = Parent(leaf=leaf, tag="ephemeral")

    with furu.overlay() as temp_ov:
        assert temp_ov.is_dir()
        assert parent.create() == "leaf:base-tmp+ephemeral"
        assert (temp_ov / "objects").is_dir()

    assert not temp_ov.exists()
    assert not parent.done
    assert leaf.done


def test_overlay_precedence_over_base(tmp_path: Path) -> None:
    """When a result exists in both overlay and base, overlay takes precedence."""
    leaf = Leaf(name="shared")
    assert leaf.create() == "leaf:shared"
    base_row = stored_row(leaf)
    assert base_row is not None

    ov = tmp_path / "ov"
    # Populate an overlay copy with a different stored payload.
    ov_run = ov / "objects" / leaf._stored_name / leaf._class_part / FIXED / "runs" / leaf._artifact_hash
    shutil.copytree(base_row.run, ov_run)
    manifest = ov_run / "result" / "manifest.json"
    manifest.write_text('[\n"from-overlay"\n]\n', encoding="utf-8")

    with furu.overlay(ov):
        assert leaf.load_existing() == "from-overlay"
    assert leaf.load_existing() == "leaf:shared"


def test_overlay_batched_reads_base_and_writes_overlay(tmp_path: Path) -> None:
    base_items = [BatchedItem(value=1), BatchedItem(value=2)]
    assert furu.create(base_items) == [10, 20]
    assert _batch_calls == [[1, 2]]

    ov = tmp_path / "ov"
    new_items = [BatchedItem(value=3), BatchedItem(value=4)]
    with _readonly_tree(*_all_base_dirs()), furu.overlay(ov):
        assert furu.create([*base_items, *new_items]) == [10, 20, 30, 40]
        assert _batch_calls == [[1, 2], [3, 4]]
        # Cached on second call inside overlay.
        assert furu.create([*base_items, *new_items]) == [10, 20, 30, 40]
        assert _batch_calls == [[1, 2], [3, 4]]

    assert all(item.done for item in base_items)
    assert not any(item.done for item in new_items)


def test_overlay_data_dir_paths_across_base_and_overlay(tmp_path: Path) -> None:
    leaf = DataLeaf(name="weights")
    base_file = leaf.create()
    assert base_file.read_text(encoding="utf-8") == "payload:weights"

    ov = tmp_path / "ov"
    parent = DataParent(leaf=leaf, suffix="v2")
    with _readonly_tree(*_all_base_dirs()), furu.overlay(ov):
        out_file = parent.create()
        assert out_file.read_text(encoding="utf-8") == "payload:weights:v2"
        assert out_file.resolve().is_relative_to(ov.resolve())
        # Base file still resolves into base.
        assert leaf.load_existing().resolve() == base_file.resolve()


# --- Custom Metadata(storage=...) --------------------------------------------


def test_overlay_custom_storage_readonly_base_and_no_basename_collision(
    tmp_path: Path,
) -> None:
    store_a = tmp_path / "a" / "results"
    store_b = tmp_path / "b" / "results"
    base_a = CustomLeaf(name="x", root=store_a)
    assert base_a.create() == "custom:x"

    ov = tmp_path / "ov"
    new_a = CustomLeaf(name="y", root=store_a)
    new_b = CustomLeaf(name="y", root=store_b)

    with _readonly_tree(store_a, *_all_base_dirs()), furu.overlay(ov):
        assert base_a.load_existing() == "custom:x"
        assert new_a.create() == "custom:y"
        assert new_b.create() == "custom:y"
        assert new_a._class_dir != new_b._class_dir
        assert new_a._class_dir.resolve().is_relative_to((ov / "custom").resolve())
        assert new_b._class_dir.resolve().is_relative_to((ov / "custom").resolve())

    # Neither custom base store was mutated.
    assert not new_a.done
    assert not store_b.exists()


# --- Spec.delete() in overlay mode -------------------------------------------


def test_overlay_delete_only_deletes_overlay_and_refuses_base(tmp_path: Path) -> None:
    base_leaf = Leaf(name="keep-me")
    base_leaf.create()
    ov_leaf = Leaf(name="throw-away")
    missing_leaf = Leaf(name="never-built")

    ov = tmp_path / "ov"
    with _readonly_tree(*_all_base_dirs()), furu.overlay(ov):
        ov_leaf.create()
        assert ov_leaf.done
        assert ov_leaf.delete(mode="force") is True
        assert not ov_leaf.done
        assert missing_leaf.delete(mode="force") is False

        with pytest.raises(RuntimeError, match="read-only base"):
            base_leaf.delete(mode="force")
        assert base_leaf.done

    assert base_leaf.done


# --- Config, FURU_OVERLAY env var, validation, nesting -----------------------


def test_config_reads_furu_overlay_env_and_anchors_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_caches: None
) -> None:
    monkeypatch.setenv("FURU_OVERLAY", str(tmp_path / "ov"))
    assert _Config().overlay == tmp_path / "ov"

    monkeypatch.setenv("FURU_OVERLAY", "   ")
    assert _Config().overlay is None

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    _project_anchor.cache_clear()
    monkeypatch.setenv("FURU_OVERLAY", "furu-overlay/exp1")
    cfg = _Config()
    assert cfg.anchored().overlay == repo.resolve() / "furu-overlay" / "exp1"


def test_overlay_rejects_overlapping_directories_and_nesting(tmp_path: Path) -> None:
    base = get_config().directories.anchored()

    # Exact base directory or inside base directory or parent of base directory.
    for bad in (base.objects, base.objects / "nested", base.objects.parent):
        with pytest.raises(ValueError, match="must not overlap"), furu.overlay(bad):
            pass

    # Nesting overlay inside overlay is refused.
    with (
        furu.overlay(tmp_path / "ov1"),
        pytest.raises(RuntimeError, match="cannot be nested"),
        furu.overlay(tmp_path / "ov2"),
    ):
        pass


class _NestedOverlaySpec(Spec):
    def create(self) -> int:
        with furu.overlay():
            return 1


def test_overlay_cannot_be_entered_inside_create() -> None:
    with pytest.raises(RuntimeError, match="cannot be entered inside create"):
        _NestedOverlaySpec().create()


# --- Code tracing (code_version="traced") ------------------------------------


def test_overlay_code_tracing_across_base_and_overlay(
    repo: Repo, tmp_path: Path
) -> None:
    repo.write(
        "pipe.py",
        """
import furu
from furu import Metadata, Spec

calls = []

class Up(Spec):
    n: int
    def metadata(self) -> Metadata:
        return Metadata(code_version="traced")
    def create(self) -> int:
        calls.append(("up", self.n))
        return self.n + 1

class Down(Spec):
    up: Up
    def metadata(self) -> Metadata:
        return Metadata(code_version="traced")
    def create(self) -> int:
        calls.append(("down", self.up.n))
        return self.up.create() * 10
""",
    )
    mod = repo.load("pipe")
    down = mod.Down(up=mod.Up(n=2))
    assert down.create() == 30
    assert mod.calls == [("up", 2), ("down", 2)]

    ov = tmp_path / "ov"
    base_dirs = _all_base_dirs()

    with _readonly_tree(*base_dirs):
        # 1. No code edits: cache hit directly from read-only base.
        with furu.overlay(ov):
            mod.calls.clear()
            assert down.create() == 30
            assert mod.calls == []

        # 2. Edit Down only: Up stays a cache hit in base, Down recomputes into overlay.
        repo.edit("pipe.py", "return self.up.create() * 10", "return self.up.create() * 100")
        mod = repo.load("pipe")
        down2 = mod.Down(up=mod.Up(n=2))
        with furu.overlay(ov):
            assert down2.create() == 300
            assert mod.calls == [("down", 2)]
            # Second run in same overlay hits overlay cache (Up in base still live).
            mod.calls.clear()
            assert down2.create() == 300
            assert mod.calls == []

        # 3. Edit Up: invalidates base Up AND overlay Down (which was built on base Up).
        repo.edit("pipe.py", "return self.n + 1", "return self.n + 5")
        mod = repo.load("pipe")
        down3 = mod.Down(up=mod.Up(n=2))
        with furu.overlay(ov):
            assert down3.create() == 700
            assert mod.calls == [("up", 2), ("down", 2)]
            # Re-running hits overlay cache (both Up and Down in overlay).
            mod.calls.clear()
            assert down3.create() == 700
            assert mod.calls == []

        # 4. Edit Up again: invalidates overlay Down built on overlay Up.
        repo.edit("pipe.py", "return self.n + 5", "return self.n + 6")
        mod = repo.load("pipe")
        down4 = mod.Down(up=mod.Up(n=2))
        with furu.overlay(ov):
            assert down4.create() == 800
            assert mod.calls == [("up", 2), ("down", 2)]

    # Revert edits and delete overlay: original base results are still live cache hits.
    shutil.rmtree(ov)
    repo.edit("pipe.py", "return self.n + 6", "return self.n + 1")
    repo.edit("pipe.py", "return self.up.create() * 100", "return self.up.create() * 10")
    mod = repo.load("pipe")
    assert mod.Down(up=mod.Up(n=2)).create() == 30
    assert mod.calls == []


def test_overlay_code_tracing_cross_storage(repo: Repo, tmp_path: Path) -> None:
    custom_store = tmp_path / "custom_up"
    repo.write(
        "cross.py",
        f"""
from pathlib import Path
from furu import Metadata, Spec

calls = []

class Up(Spec):
    n: int
    def metadata(self) -> Metadata:
        return Metadata(storage=Path({str(custom_store)!r}), code_version="traced")
    def create(self) -> int:
        calls.append(("up", self.n))
        return self.n + 1

class Down(Spec):
    up: Up
    def metadata(self) -> Metadata:
        return Metadata(code_version="traced")
    def create(self) -> int:
        calls.append(("down", self.up.n))
        return self.up.create() * 2
""",
    )
    mod = repo.load("cross")
    assert mod.Up(n=3).create() == 4
    assert mod.calls == [("up", 3)]

    ov = tmp_path / "ov"
    with _readonly_tree(custom_store, *_all_base_dirs()), furu.overlay(ov):
        mod.calls.clear()
        down = mod.Down(up=mod.Up(n=3))
        assert down.create() == 8
        assert mod.calls == [("down", 3)]
        mod.calls.clear()
        assert down.create() == 8
        assert mod.calls == []


# --- Migrations / schema.ndjson / ask() --------------------------------------


def test_overlay_schema_checks_base_readonly_and_writes_overlay_only(
    repo: Repo, tmp_path: Path
) -> None:
    repo.write(
        "mig.py",
        """
from furu import Spec

class Model(Spec):
    name: str
    def create(self) -> str:
        return self.name
""",
    )
    mod = repo.load("mig")
    assert mod.Model(name="a").create() == "a"
    base_schema = schema_path_in(mod.Model(name="a")._base_class_dir)
    base_schema_before = base_schema.read_text(encoding="utf-8")

    # Add a new field with Added(default=1). Base schema.ndjson has never seen it.
    repo.write(
        "mig.py",
        """
from furu import Added, Spec

class Model(Spec):
    migrations = (Added("extra", default=1),)
    name: str
    extra: int = 1
    def create(self) -> str:
        return f"{self.name}:{self.extra}"
""",
    )
    mod = repo.load("mig")
    ov = tmp_path / "ov"

    with _readonly_tree(*_all_base_dirs()), furu.overlay(ov):
        # Pure cache hit from base does NOT lock/write base or even create overlay.
        assert mod.Model(name="a", extra=1).load_existing() == "a"
        assert not (ov / "objects").exists()

        # Creating a new spec writes schema.ndjson ONLY to overlay.
        assert mod.Model(name="b", extra=2).create() == "b:2"
        ov_schema = schema_path_in(mod.Model(name="b", extra=2)._class_dir)
        assert ov_schema.is_file()

    assert base_schema.read_text(encoding="utf-8") == base_schema_before

    # Changing the default in the SAME overlay is caught against overlay's schema.ndjson.
    repo.write(
        "mig.py",
        """
from furu import Added, Spec

class Model(Spec):
    migrations = (Added("extra", default=99),)
    name: str
    extra: int = 99
    def create(self) -> str:
        return f"{self.name}:{self.extra}"
""",
    )
    mod = repo.load("mig")
    with (
        _readonly_tree(*_all_base_dirs()),
        furu.overlay(ov),
        pytest.raises(MigrationError, match="recorded 1"),
    ):
        mod.Model(name="b", extra=99).create()

    # After deleting overlay, base never recorded "extra", so default=99 works cleanly.
    shutil.rmtree(ov)
    with _readonly_tree(*_all_base_dirs()), furu.overlay(ov):
        assert mod.Model(name="c", extra=99).create() == "c:99"


def test_overlay_schema_catches_conflict_with_base_recorded_default(
    repo: Repo, tmp_path: Path
) -> None:
    repo.write(
        "mig2.py",
        """
from furu import Added, Spec

class Model(Spec):
    migrations = (Added("extra", default=1),)
    name: str
    extra: int = 1
    def create(self) -> str:
        return f"{self.name}:{self.extra}"
""",
    )
    mod = repo.load("mig2")
    assert mod.Model(name="a", extra=1).create() == "a:1"

    # Edit default to 2 and run in overlay against read-only base.
    repo.write(
        "mig2.py",
        """
from furu import Added, Spec

class Model(Spec):
    migrations = (Added("extra", default=2),)
    name: str
    extra: int = 2
    def create(self) -> str:
        return f"{self.name}:{self.extra}"
""",
    )
    mod = repo.load("mig2")
    with (
        _readonly_tree(*_all_base_dirs()),
        furu.overlay(tmp_path / "ov"),
        pytest.raises(MigrationError, match="recorded 1"),
    ):
        _ = mod.Model(name="a", extra=2).done


def test_overlay_ask_checks_unexplained_base_folders(
    repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.write(
        "unexp.py",
        """
from furu import Spec

class Stage(Spec):
    x: int
    def create(self) -> int:
        return self.x
""",
    )
    mod = repo.load("unexp")
    old_spec = mod.Stage(x=1)
    assert old_spec.create() == 1
    old_folder_name = f"{old_spec._stored_name}:{old_spec._class_part}"

    # Add a field without a migration step.
    repo.write(
        "unexp.py",
        """
from furu import Spec

class Stage(Spec):
    x: int
    y: int = 0
    def create(self) -> int:
        return self.x + self.y
""",
    )
    mod = repo.load("unexp")
    ov = tmp_path / "ov"

    with _readonly_tree(*_all_base_dirs()), furu.overlay(ov):
        with pytest.raises(MigrationError, match="leave stored results behind"):
            mod.Stage(x=1, y=2).create()

        # Respects FURU_IGNORE_STORED without touching base.
        monkeypatch.setenv("FURU_IGNORE_STORED", old_folder_name)
        assert mod.Stage(x=1, y=2).create() == 3


# --- @furu.dependency(remember=True) -----------------------------------------


def test_overlay_remember_reads_base_and_writes_overlay(tmp_path: Path) -> None:
    base_sweep = RememberSweep(count=2)
    assert base_sweep.create() == ["leaf:r0", "leaf:r1"]
    assert _remember_calls == [2]
    assert remembered_dir_in(base_sweep._base_class_dir).is_dir()

    ov = tmp_path / "ov"
    ov_sweep = RememberSweep(count=3)
    with _readonly_tree(*_all_base_dirs()), furu.overlay(ov):
        _remember_calls.clear()
        # Re-create fresh instance to test remembered lookup from base.
        assert RememberSweep(count=2).load_existing() == ["leaf:r0", "leaf:r1"]
        assert _remember_calls == []

        # New count runs dependency method and writes remembered only to overlay.
        assert ov_sweep.create() == ["leaf:r0", "leaf:r1", "leaf:r2"]
        assert _remember_calls == [3]
        assert remembered_dir_in(ov_sweep._class_dir).is_dir()

    assert not ov_sweep.done


# --- Provenance & git checks -------------------------------------------------


def test_overlay_git_checks_local_vs_snapshot(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_caches: None,
) -> None:
    (git_repo / "pyproject.toml").write_text("[project]\nname = 't'\nversion = '0.1.0'\n")
    (git_repo / "uv.lock").write_text("version = 1\n")
    commit_and_push(git_repo, "project")
    monkeypatch.chdir(git_repo)
    monkeypatch.setattr(provenance, "_require_pushed_code", True)

    base_dirs = _FuruDirectories(
        **{name: git_repo / "furu-data" / name for name in _FuruDirectories.model_fields}
    )
    ov = tmp_path / "ov"

    # Make working tree dirty.
    (git_repo / "tracked.txt").write_text("dirty\n")

    with override_config(_Config(directories=base_dirs)):
        # Overlay mode + local (snapshot=False) allows dirty & unpushed edits,
        # and never creates or touches base furu-data/.
        with furu.overlay(ov):
            captured = provenance.capture_submit_provenance(snapshot=False)
            assert captured.git.repo_root == str(git_repo.resolve())
            assert not (git_repo / "furu-data").exists()

            # Overlay mode + Slurm (snapshot=True) rejects dirty tree (no push needed).
            with pytest.raises(RuntimeError, match="no push needed in overlay mode"):
                provenance.capture_submit_provenance(snapshot=True)
            # Never touched base furu-data/ with .gitignore.
            assert not (git_repo / "furu-data").exists()

        # Normal mode rejects dirty working tree.
        with pytest.raises(RuntimeError, match="uncommitted changes"):
            provenance.capture_submit_provenance()
        shutil.rmtree(git_repo / "furu-data")

        # Commit locally WITHOUT pushing to origin.
        git(git_repo, "commit", "-qam", "local wip")
        local_commit = git(git_repo, "rev-parse", "HEAD")

        # Overlay mode + Slurm (snapshot=True) ALLOWS unpushed local commit and
        # extracts snapshot into overlay/snapshots/, never base/snapshots/.
        with furu.overlay(ov):
            captured = provenance.capture_submit_provenance(snapshot=True)
            assert captured.git.commit == local_commit
            snap = extract_snapshot(local_commit, repo=git_repo)
            assert snap == ov / "snapshots" / local_commit
            assert (snap / "tracked.txt").read_text() == "dirty\n"
            assert not (git_repo / "furu-data").exists()

        # Normal mode still rejects unpushed commit.
        with pytest.raises(RuntimeError, match="not on any branch of origin"):
            provenance.capture_submit_provenance()


# --- Workers & shared Manager isolation --------------------------------------


def test_overlay_with_local_thread_workers_and_dynamic_deps(tmp_path: Path) -> None:
    base_leaf = Leaf(name="worker-base")
    assert base_leaf.create() == "leaf:worker-base"

    ov = tmp_path / "ov"
    parent = Parent(leaf=base_leaf, tag="w1")
    dyn = DynamicParent(name="worker-new", tag="w2")

    base_dirs = _all_base_dirs()
    before = {str(d): _snapshot_tree(d) for d in base_dirs}

    with _readonly_tree(*base_dirs), furu.overlay(ov):
        results = furu.create(
            [parent, dyn], on=[LocalThreadWorkerBackend(max_workers=2)]
        )
        assert results == ["leaf:worker-base+w1", "dyn:leaf:worker-new+w2"]
        assert list((ov / "executions").iterdir())
        assert list((ov / "plans").glob("*.plan"))
        assert list((ov / "provenance").iterdir())

    assert {str(d): _snapshot_tree(d) for d in base_dirs} == before
    assert base_leaf.done
    assert not parent.done
    assert not dyn.done


def test_shared_manager_isolates_different_overlays_and_dedups_same_overlay(
    tmp_path: Path,
) -> None:
    leaf = Leaf(name="shared-mgr")
    backend = LocalThreadWorkerBackend(max_workers=1)
    pool_key = backend.pool_key
    manager = Manager(url="ws://unused", manager_dir=None)

    def submit_with(cfg: _Config, plan_name: str) -> tuple[Any, Run]:
        with override_config(cfg):
            run = Run(
                backends={pool_key: backend},
                launch=Launch(
                    plan=tmp_path / f"{plan_name}.plan",
                    execution_log=tmp_path / f"{plan_name}.log",
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
            groups = run.place([leaf])
            lines = list(map(plan.dump, plan.lines([leaf], {leaf.object_id})))
            submit = run.submit(groups, lines, tmp_path / f"{plan_name}.plan")
        from furu.execution.manager import _Script

        script = _Script(MagicMock())
        manager.scripts.append(script)
        manager._add(script, submit)
        return script, run

    base_cfg = get_config()
    ov1 = tmp_path / "ov1"
    ov2 = tmp_path / "ov2"

    # Real + ov1 + ov2 + duplicate ov1: 3 distinct jobs (real, ov1, ov2),
    # while the two ov1 scripts share one job.
    s_real, _ = submit_with(base_cfg, "real")
    s_ov1_a, _ = submit_with(base_cfg.model_copy(update={"overlay": ov1}), "ov1-a")
    s_ov1_b, _ = submit_with(base_cfg.model_copy(update={"overlay": ov1}), "ov1-b")
    s_ov2, _ = submit_with(base_cfg.model_copy(update={"overlay": ov2}), "ov2")

    leases = []
    while (lease := manager._hand_out(MagicMock(), pool_key)) is not None:
        leases.append(lease)

    assert len(leases) == 3
    assert [lease.job.config.overlay for lease in leases] == [None, ov1, ov2]
    assert [lease.job.specs for lease in leases] == [
        [leaf.object_id],
        [leaf.object_id],
        [leaf.object_id],
    ]
    assert leases[0].job.class_dir == leaf._base_class_dir
    assert leases[1].job.class_dir.resolve().is_relative_to(ov1.resolve())
    assert leases[2].job.class_dir.resolve().is_relative_to(ov2.resolve())

    # Finishing ov1's lease notifies BOTH ov1 scripts and neither of the others.
    manager._finish(leases[1], JobCompletedResult())
    notified = [script for script, _ in manager.outbox]
    assert notified == [s_ov1_a, s_ov1_b]
    assert all(
        msg.object_ids == [leaf.object_id]
        for _, msg in manager.outbox
        if hasattr(msg, "object_ids")
    )
    # Clean up remaining leases.
    manager.outbox.clear()
    manager._finish(leases[0], JobCompletedResult())
    assert [script for script, _ in manager.outbox] == [s_real]
    manager.outbox.clear()
    manager._finish(leases[2], JobCompletedResult())
    assert [script for script, _ in manager.outbox] == [s_ov2]


# --- Nested dataclass migration in overlay -----------------------------------


@dataclass(frozen=True)
class _Cfg:
    a: int


class _UsesCfg(Spec):
    cfg: _Cfg

    def create(self) -> int:
        return self.cfg.a


def test_overlay_records_nested_names_without_touching_base(tmp_path: Path) -> None:
    base_obj = _UsesCfg(cfg=_Cfg(a=1))
    assert base_obj.create() == 1
    ov = tmp_path / "ov"
    with _readonly_tree(*_all_base_dirs()), furu.overlay(ov):
        new_obj = _UsesCfg(cfg=_Cfg(a=2))
        assert new_obj.create() == 2
        assert schema_path_in(new_obj._class_dir).is_file()
    assert not _UsesCfg(cfg=_Cfg(a=2)).done


# --- Subprocess env var end-to-end -------------------------------------------


def test_furu_overlay_env_var_end_to_end_subprocess(tmp_path: Path) -> None:
    leaf = Leaf(name="env-base")
    assert leaf.create() == "leaf:env-base"
    base = get_config().directories.anchored()
    ov = tmp_path / "ov"

    code = """
import furu
from test_overlay import Leaf, Parent

base_leaf = Leaf(name="env-base")
assert base_leaf.done
assert base_leaf.load_existing() == "leaf:env-base"
p = Parent(leaf=base_leaf, tag="from-env")
assert p.create() == "leaf:env-base+from-env"
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
    with _readonly_tree(*_all_base_dirs()):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    assert result.stdout.strip().endswith("OK")
    assert not Parent(leaf=leaf, tag="from-env").done
    assert (ov / "objects").is_dir()
