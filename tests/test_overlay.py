"""A debug session: FURU_OVERLAY reads the store and writes under one folder."""

import os
import shutil
import subprocess
import sys
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from gitrepo import git
from runs import run_of
from tracedrepo import Repo, fresh_process

import furu
from furu import Added, Spec, config
from furu.config import _Config, get_config, layers, over, under
from furu.execution.launch import ask
from furu.migration.steps import MigrationError
from furu.provenance import GitIdentity
from furu.snapshot import extract_snapshot, require_pushed
from furu.storage._layout import (
    deleted_dir_in,
    remembered_dir_in,
    schema_path_in,
    trace_path_in,
)
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend

# The create() and dependency methods that ran in this process, by name.
_calls: list[str] = []


class Data(Spec):
    name: str

    def create(self) -> str:
        _calls.append(self.name)
        return f"data:{self.name}"


class Model(Spec):
    data: Data
    lr: float

    def create(self) -> str:
        _calls.append(f"model:{self.lr}")
        return f"{self.data.create()}@{self.lr}"


class Elsewhere(Spec):
    name: str
    root: str

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(storage=Path(self.root))

    def create(self) -> str:
        return self.name


class Sweep(Spec):
    name: str
    size: int

    @furu.dependency(remember=True)
    def runs(self) -> list[Data]:
        _calls.append(f"runs:{self.size}")
        return [Data(name=f"{self.name}{index}") for index in range(self.size)]

    def create(self) -> int:
        return len(furu.load_existing(self.runs))


PLOTS = """\
import furu

SCALE = 2


class Curve(furu.Spec):
    name: str

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return SCALE


class Plot(furu.Spec):
    curve: Curve

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> str:
        return f"plot:{self.curve.create()}"
"""


@pytest.fixture(autouse=True)
def _a_fresh_process() -> None:
    _calls.clear()
    fresh_process()


@pytest.fixture
def overlay(tmp_path: Path) -> Path:
    return tmp_path / "overlay"


@contextmanager
def session(root: Path) -> Generator[None]:
    """What a script started with FURU_OVERLAY=root sees."""
    fresh_process()
    with override_config(get_config().model_copy(update={"overlay": root})):
        yield
    fresh_process()


def _tree(root: Path) -> dict[str, tuple[int, int, int]]:
    """Every path at and under root, with what a write to it would change."""
    return {
        str(path): (stat.st_mode, stat.st_size, stat.st_mtime_ns)
        for path in [root, *sorted(root.rglob("*"))]
        for stat in [path.lstat()]
    }


@contextmanager
def read_only_store() -> Generator[None]:
    """Nothing in the directories furu is configured with can be written, and
    nothing was by the end."""
    roots = [path for _, path in get_config().directories.anchored()]
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    paths = [path for root in roots for path in (root, *root.rglob("*"))]
    modes = {path: path.stat().st_mode for path in paths if not path.is_symlink()}
    for path, mode in modes.items():
        path.chmod(mode & ~0o222)
    before = {root: _tree(root) for root in roots}
    try:
        with pytest.raises(PermissionError):
            (roots[0] / "written").touch()
        yield
        assert {root: _tree(root) for root in roots} == before
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _merge(**body: Any) -> Callable[..., Spec]:
    """The spec class Merge, as its source reads at one point in time."""
    body = {"__module__": __name__, "create": lambda self: "merged", **body}
    return types.new_class("Merge", (Spec,), exec_body=lambda ns: ns.update(body))


def _merge_with_overlap(default: float) -> Callable[..., Spec]:
    return _merge(
        __annotations__={"lemma": str, "min_overlap": float},
        min_overlap=default,
        migrations=(Added("min_overlap", default=default),),
    )


# --- where a session reads and writes ---


def test_a_path_is_mirrored_under_the_overlay_and_back(overlay: Path) -> None:
    store = Path("/data/furu/objects")

    assert over(store) == under(store) == store
    assert layers(store) == [store]

    with session(overlay):
        assert over(store) == overlay / "data/furu/objects"
        assert over(over(store)) == over(store)
        assert under(over(store)) == under(store) == store
        assert layers(store) == layers(over(store)) == [over(store), store]
        assert over(Path("/scratch/objects")) != over(Path("/data/objects"))


def test_every_directory_and_the_manager_are_under_the_overlay(overlay: Path) -> None:
    stores = dict(get_config().directories.anchored())
    stores_manager = get_config().manager_version_dir

    with session(overlay):
        directories = get_config().directories.anchored()
        for name, path in directories:
            assert path.is_relative_to(overlay)
            assert under(path) == stores[name]
        manager = get_config().manager_version_dir
        assert manager.is_relative_to(overlay)
        assert under(manager) == stores_manager
        # A worker is given paths that are under the overlay already.
        assert get_config().anchored().directories == directories


def test_a_session_reads_the_store_and_writes_under_the_overlay(overlay: Path) -> None:
    Data(name="mnist").create()
    stored = run_of(Data(name="mnist"))
    _calls.clear()

    with read_only_store(), session(overlay):
        data = Data(name="mnist")
        assert data.done
        assert data.load_existing() == "data:mnist"
        assert run_of(data) == stored

        model = Model(data=data, lr=0.1)
        assert not model.done
        assert model.create() == "data:mnist@0.1"
        assert run_of(model).is_relative_to(overlay)
        assert under(run_of(model)).parents[4] == stored.parents[4]
        assert model.done

    assert _calls == ["model:0.1"]
    assert not Model(data=Data(name="mnist"), lr=0.1).done
    assert Data(name="mnist").done


def test_deleting_the_overlay_undoes_the_session(overlay: Path) -> None:
    with session(overlay):
        assert Data(name="mnist").create() == "data:mnist"
        assert Data(name="mnist").done
    shutil.rmtree(overlay)
    config._claimed.cache_clear()  # as in the next process

    with session(overlay):
        assert not Data(name="mnist").done
    assert not Data(name="mnist").done


def test_the_overlays_result_wins_over_the_stores(overlay: Path) -> None:
    with session(overlay):
        Data(name="mnist").create()
    Data(name="mnist").create()
    assert _calls == ["mnist", "mnist"]
    stored = run_of(Data(name="mnist"))

    with session(overlay):
        assert run_of(Data(name="mnist")).is_relative_to(overlay)
        assert under(run_of(Data(name="mnist"))) == stored
    assert run_of(Data(name="mnist")) == stored


def test_a_storage_folder_of_a_spec_is_mirrored_too(
    overlay: Path, tmp_path: Path
) -> None:
    roots = [str(tmp_path / "fast"), str(tmp_path / "slow")]
    Elsewhere(name="a", root=roots[0]).create()

    with session(overlay):
        fast, slow = (Elsewhere(name="a", root=root) for root in roots)
        assert run_of(fast).is_relative_to(roots[0])
        assert slow.create() == "a"
        assert run_of(slow).is_relative_to(over(Path(roots[1])))

    assert not Path(roots[1]).exists()


def test_workers_of_a_session_write_under_the_overlay(overlay: Path) -> None:
    furu.build([Data(name="mnist")], on=[LocalThreadWorkerBackend()])

    with read_only_store(), session(overlay):
        models = [Model(data=Data(name=name), lr=0.1) for name in ("mnist", "cifar")]
        furu.build(models, on=[LocalThreadWorkerBackend()])

        assert [model.load_existing() for model in models] == [
            "data:mnist@0.1",
            "data:cifar@0.1",
        ]
        assert not run_of(Data(name="mnist")).is_relative_to(overlay)
        for spec in (*models, Data(name="cifar")):
            assert run_of(spec).is_relative_to(overlay)
        directories = get_config().directories.anchored()
        assert list(directories.plans.glob("*.plan"))
        assert _files(directories.provenance)

    assert not Data(name="cifar").done


# --- which folder is an overlay ---


def test_a_new_folder_is_marked_as_an_overlay(overlay: Path) -> None:
    with session(overlay):
        assert not overlay.exists()  # nothing is written until furu writes
        Data(name="mnist").create()

    assert (overlay / ".gitignore").read_text().endswith("\n*\n")
    assert "FURU_OVERLAY" in (overlay / ".gitignore").read_text()

    config._claimed.cache_clear()  # as in the next process
    with session(overlay):
        assert Data(name="mnist").done
        assert Data(name="cifar").create() == "data:cifar"


def test_a_folder_with_other_files_is_refused_and_left_alone(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("*.pyc\n")
    before = _tree(tmp_path)

    with session(tmp_path):
        with pytest.raises(RuntimeError, match="not a furu overlay's"):
            Data(name="mnist").create()
        with pytest.raises(RuntimeError, match="not a furu overlay's"):
            _ = Data(name="mnist").done

    assert _tree(tmp_path) == before


def test_a_relative_overlay_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with (
        session(Path("debug")),
        pytest.raises(RuntimeError, match="FURU_OVERLAY=debug is not an absolute"),
    ):
        Data(name="mnist").create()

    assert not (tmp_path / "debug").exists()


def test_an_overlay_in_the_repo_keeps_the_checkout_clean(git_repo: Path) -> None:
    with session(git_repo / "debug"):
        Data(name="mnist").create()
        require_pushed(GitIdentity.capture(git_repo))

        for _, directory in get_config().directories.anchored():
            assert (directory / ".gitignore").is_file()
        commit = git(git_repo, "rev-parse", "HEAD")
        code_dir = extract_snapshot(commit, repo=git_repo)
        assert code_dir.is_relative_to(git_repo / "debug")
        assert (code_dir / "tracked.txt").read_text() == "content\n"

    assert git(git_repo, "status", "--porcelain") == ""


# --- only the environment asks for a session ---


@pytest.mark.usefixtures("fresh_caches")
def test_the_environment_asks_for_an_overlay(
    overlay: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert config._load().overlay is None

    monkeypatch.setenv("FURU_OVERLAY", str(overlay))
    assert config._load().overlay == overlay

    monkeypatch.setenv("FURU_OVERLAY", "")
    assert config._load().overlay is None
    assert not overlay.exists()


@pytest.mark.usefixtures("fresh_caches")
@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("pyproject.toml", '[tool.furu]\noverlay = "/tmp/debug"\n'),
        (".env", "FURU_OVERLAY=/tmp/debug\n"),
    ],
)
def test_a_file_that_asks_for_an_overlay_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, text: str
) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / name).write_text(text)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="FURU_OVERLAY=/tmp/debug"):
        config._load()


def test_a_worker_is_told_its_overlay(overlay: Path) -> None:
    with session(overlay):
        sent = get_config().anchored().model_dump_json()

    assert _Config.model_validate_json(sent).overlay == overlay


def test_pytest_runs_outside_an_overlay_asked_for_in_the_shell(
    tmp_path: Path, overlay: Path
) -> None:
    test = tmp_path / "test_isolated.py"
    test.write_text(
        "import os\n"
        "import furu\n"
        "from furu.config import get_config\n"
        "class Leaf(furu.Spec):\n"
        "    def create(self) -> int:\n"
        "        return 1\n"
        "def test_isolated():\n"
        "    assert 'FURU_OVERLAY' not in os.environ\n"
        "    assert get_config().overlay is None\n"
        "    assert not Leaf().done\n"
    )

    ran = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test)],
        cwd=tmp_path,
        check=False,
        env={**os.environ, "FURU_OVERLAY": str(overlay)},
        capture_output=True,
        text=True,
    )

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not overlay.exists()


# --- deleting ---


def test_deleting_in_a_session_leaves_the_store_and_computes_again(
    overlay: Path,
) -> None:
    Data(name="mnist").create()
    stored = run_of(Data(name="mnist"))
    _calls.clear()

    with read_only_store(), session(overlay):
        assert Data(name="mnist").delete("force")
        assert stored.is_dir()
        whiteouts = deleted_dir_in(Data(name="mnist")._class_dir)
        assert [path.name for path in whiteouts.iterdir()] == [stored.name]

        assert not Data(name="mnist").done
        with pytest.raises(furu.Missing):
            Data(name="mnist").load_existing()
        assert not Data(name="mnist").delete("force")

        assert Data(name="mnist").create() == "data:mnist"
        assert run_of(Data(name="mnist")).is_relative_to(overlay)
        assert _calls == ["mnist"]

        # The next delete removes the overlay's run, and the store's stays out.
        assert Data(name="mnist").delete("force")
        assert not Data(name="mnist").done
        assert not Data(name="mnist").delete("force")

    assert run_of(Data(name="mnist")) == stored


def test_what_was_built_on_a_deleted_result_stays_done(overlay: Path) -> None:
    Model(data=Data(name="mnist"), lr=0.1).create()

    with session(overlay):
        assert Data(name="mnist").delete("force")

        assert not Data(name="mnist").done
        assert Model(data=Data(name="mnist"), lr=0.1).done


def test_delete_asks_before_it_stops_counting_a_run_of_the_store(
    overlay: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Data(name="mnist").create()
    stored = run_of(Data(name="mnist"))
    asked: list[str] = []
    answers = iter(["n", "y"])

    def answer(prompt: str) -> str:
        asked.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)

    with session(overlay):
        assert not Data(name="mnist").delete()
        assert Data(name="mnist").done
        assert Data(name="mnist").delete()
        assert not Data(name="mnist").done

    prompt = (
        f"Do you want to stop counting {stored}, which this debug session "
        "leaves in the store? [y/N] "
    )
    assert asked == [prompt, prompt]


# --- schema, remembered lists and traced code ---


def test_a_session_records_schema_lines_in_the_overlay(overlay: Path) -> None:
    before = _merge(__annotations__={"lemma": str})
    before(lemma="comer").create()
    stored = schema_path_in(before(lemma="comer")._class_dir)
    lines = stored.read_text()

    with read_only_store(), session(overlay):
        after = _merge_with_overlap(0.5)(lemma="comer")
        assert after.done
        assert after.load_existing() == "merged"

        ours = schema_path_in(after._class_dir)
        assert ours.is_relative_to(overlay)
        assert '"added":"min_overlap"' in ours.read_text().replace(" ", "")
        with pytest.raises(MigrationError, match="recorded 0.5"):
            _ = _merge_with_overlap(0.8)(lemma="comer").done

    assert stored.read_text() == lines
    # Outside the session nothing recorded a default yet.
    assert _merge_with_overlap(0.8)(lemma="comer").done


def test_a_default_the_store_recorded_counts_in_a_session(overlay: Path) -> None:
    _merge_with_overlap(0.5)(lemma="comer").create()

    with read_only_store(), session(overlay):
        assert _merge_with_overlap(0.5)(lemma="comer").done
        with pytest.raises(MigrationError, match="recorded 0.5"):
            _ = _merge_with_overlap(0.8)(lemma="comer").done
        with pytest.raises(MigrationError, match="recorded 0.5"):
            _merge_with_overlap(0.8)(lemma="comer").create()


def test_a_folder_of_the_store_that_the_class_does_not_explain_stops_a_session(
    overlay: Path,
) -> None:
    _merge(__annotations__={"lemma": str})(lemma="comer").create()
    renamed = _merge(__annotations__={"word": str})(word="comer")

    with (
        read_only_store(),
        session(overlay),
        pytest.raises(MigrationError, match="this launch stores in"),
    ):
        ask([renamed])


def test_a_session_reuses_and_adds_to_what_the_store_remembers(overlay: Path) -> None:
    assert furu.create(Sweep(name="a", size=2)) == 2
    stored = _files(remembered_dir_in(Sweep(name="a", size=2)._class_dir))
    assert stored and _calls.count("runs:2") == 1

    with read_only_store(), session(overlay):
        # The id is taken from the store's line, without a call of the method.
        assert furu.create(Sweep(name="a", size=2)) == 2
        assert _calls.count("runs:2") == 1

        assert furu.create(Sweep(name="b", size=1)) == 1
        ours = remembered_dir_in(Sweep(name="b", size=1)._class_dir)
        assert ours.is_relative_to(overlay)
        assert _files(ours)
        assert _calls.count("runs:1") == 1

    with session(overlay):
        assert furu.create(Sweep(name="b", size=1)) == 1
        assert _calls.count("runs:1") == 1
    assert _files(remembered_dir_in(Sweep(name="a", size=2)._class_dir)) == stored


@pytest.fixture
def plots(repo: Repo) -> ModuleType:
    repo.write("plots.py", PLOTS)
    return repo.load("plots")


def _version(spec: Spec) -> Path:
    return run_of(spec).parent.parent


def test_a_session_runs_edited_traced_code_and_keeps_the_stores_results(
    repo: Repo, plots: ModuleType, overlay: Path
) -> None:
    assert plots.Plot(curve=plots.Curve(name="a")).create() == "plot:2"
    stored = _version(plots.Plot(curve=plots.Curve(name="a")))

    with read_only_store(), session(overlay):
        plot = plots.Plot(curve=plots.Curve(name="a"))
        assert plot.done
        assert _version(plot) == stored

        # Another spec of unchanged code builds on the store's version by the
        # name the store has for it.
        other = plots.Plot(curve=plots.Curve(name="b"))
        assert other.create() == "plot:2"
        assert _version(other).is_relative_to(overlay)
        assert _version(other).name == stored.name
        assert trace_path_in(_version(other)).read_text() == (
            trace_path_in(stored).read_text()
        )

        repo.edit("plots.py", "SCALE = 2", "SCALE = 3")
        plots = repo.load("plots")
        edited = plots.Plot(curve=plots.Curve(name="a"))
        assert not edited.done
        assert not edited.curve.done
        assert edited.create() == "plot:3"
        for spec in (edited, edited.curve):
            assert _version(spec).is_relative_to(overlay)
            assert _version(spec).name != stored.name
        built_on = _version(edited.curve)
        assert f'"{edited.curve._stored_name}/{edited.curve._class_part}/' in (
            trace_path_in(_version(edited)).read_text()
        )
        assert built_on.name in trace_path_in(_version(edited)).read_text()
        assert str(overlay) not in trace_path_in(_version(edited)).read_text()

    # The store still holds what the committed code made, for that code.
    repo.edit("plots.py", "SCALE = 3", "SCALE = 2")
    plots = repo.load("plots")
    assert plots.Plot(curve=plots.Curve(name="a")).load_existing() == "plot:2"
    assert not plots.Plot(curve=plots.Curve(name="b")).done
