"""An overlay: a debug session that loads the real results and writes only to its own folder."""

import shutil
import types
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import pytest
from runs import run_of
from subprocess_objects import SubprocessEnvLeaf
from tracedrepo import Repo, fresh_process

import furu
from furu import Added, Spec
from furu.config import _Config, _project_anchor, get_config
from furu.migration.steps import MigrationError
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend


class Leaf(Spec):
    n: int
    calls: ClassVar[list[int]] = []

    def create(self) -> int:
        Leaf.calls.append(self.n)
        return self.n


class Double(Spec):
    leaf: Leaf

    def create(self) -> int:
        return self.leaf.create() * 2


@contextmanager
def _overlay(path: Path) -> Generator[None]:
    with override_config(get_config().model_copy(update={"overlay": path})):
        yield


def _real() -> dict[Path, int]:
    """Every file and folder under the real data directories, with its mtime."""
    return {
        path: path.stat().st_mtime_ns
        for _, directory in get_config().directories.anchored()
        if directory.exists()
        for path in [directory, *directory.rglob("*")]
    }


def test_overlay_loads_real_results_and_writes_only_to_itself(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    Leaf.calls.clear()
    assert Leaf(n=1).create() == 1
    before = _real()

    with _overlay(overlay):
        assert Double(leaf=Leaf(n=1)).create() == 2
        assert Leaf.calls == [1]  # loaded from the real store
        assert not run_of(Leaf(n=1)).is_relative_to(overlay)
        assert run_of(Double(leaf=Leaf(n=1))).is_relative_to(overlay)
        plans = get_config().directories.anchored().plans
        assert list(get_config().written(plans).iterdir())

    assert _real() == before
    assert not Double(leaf=Leaf(n=1)).done

    shutil.rmtree(overlay)
    with _overlay(overlay):
        assert not Double(leaf=Leaf(n=1)).done
        assert Leaf(n=1).done


def test_overlay_reaches_worker_processes(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    before = _real()

    with _overlay(overlay):
        leaf = SubprocessEnvLeaf(variable_name="FURU_TEST", variable_value="x")
        assert furu.create(leaf, on=(LocalThreadWorkerBackend(),)).endswith(":x")
        assert run_of(leaf).is_relative_to(overlay)

    assert _real() == before


def test_delete_under_an_overlay_leaves_real_results_alone(tmp_path: Path) -> None:
    Leaf(n=1).create()

    with _overlay(tmp_path / "overlay"):
        with pytest.raises(RuntimeError, match="outside the overlay"):
            Leaf(n=1).delete("force")
        Leaf(n=2).create()
        assert Leaf(n=2).delete("force")
        assert not Leaf(n=2).done

    assert Leaf(n=1).done


def _video(default: int) -> Spec:
    cls = types.new_class(
        "Video",
        (Spec,),
        exec_body=lambda ns: ns.update(
            __module__=__name__,
            __annotations__={"clip": str, "fps": int},
            fps=default,
            migrations=(Added("fps", default=default),),
            create=lambda self: self.fps,
        ),
    )
    return cls(clip="x")


def test_overlay_checks_defaults_the_real_store_recorded(tmp_path: Path) -> None:
    assert _video(30).create() == 30

    with _overlay(tmp_path / "overlay"):
        with pytest.raises(MigrationError, match="recorded 30"):
            _ = _video(25).done
        with pytest.raises(MigrationError, match="recorded 30"):
            _video(25).create()
        assert _video(30).done


CHILD = """\
import furu


class Child(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return self.n + 1
"""

PARENT = """\
import furu

from PKG.child import Child


class Parent(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return Child(n=self.n).create() * 10
"""


def test_traced_results_built_on_real_ones_follow_their_code(
    repo: Repo, tmp_path: Path
) -> None:
    overlay = tmp_path / "overlay"
    repo.write("child.py", CHILD)
    repo.write("parent.py", PARENT)
    parent = repo.load("parent")
    parent.Child(n=1).create()
    before = _real()

    with _overlay(overlay):
        assert parent.Parent(n=1).create() == 20
        assert not run_of(parent.Child(n=1)).is_relative_to(overlay)
        fresh_process()
        assert parent.Parent(n=1).done

        repo.edit("child.py", "self.n + 1", "self.n + 2")
        parent = repo.load("parent")
        assert not parent.Parent(n=1).done
        assert parent.Parent(n=1).create() == 30
        assert run_of(parent.Child(n=1)).is_relative_to(overlay)

    assert _real() == before


def test_overlay_comes_from_the_environment_and_anchors_like_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURU_OVERLAY", "debug")
    config = _Config()
    overlay = _project_anchor() / "debug"
    assert config.anchored().overlay == overlay

    real = Path("/data/objects/Leaf")
    assert config.written(real) == overlay / "data" / "objects" / "Leaf"
    assert config.layers(config.written(real)) == [config.written(real), real]
    assert config.layers(real) == [real]
