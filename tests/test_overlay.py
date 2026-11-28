"""``FURU_OVERLAY``: a session that reads the real store and writes only to its overlay."""

import json
import os
import shutil
import subprocess
import sys
import types
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from gitrepo import git
from runs import run_of
from tracedrepo import Repo, fresh_process

import furu
from furu import Added, Metadata, Spec, batched
from furu.config import _claim, _Config, get_config, layers, lower, upper
from furu.core import attempt_parent
from furu.migration.steps import MigrationError
from furu.provenance import GitIdentity
from furu.snapshot import require_pushed
from furu.storage._layout import remembered_dir_in, schema_path_in, trace_path_in
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend

# What Echo.create() returns now, and the argument of each Sweep.parts call.
_says = ["real"]
_calls: list[int] = []


class Corpus(Spec):
    name: str

    def create(self) -> str:
        return f"corpus {self.name}"


class Tokens(Spec):
    corpus: Corpus

    def create(self) -> str:
        return f"{self.corpus.create().upper()}|{os.getpid()}"


class Echo(Spec):
    name: str

    def create(self) -> str:
        return _says[0]


class Elsewhere(Spec):
    root: str
    name: str

    def metadata(self) -> Metadata:
        return Metadata(storage=Path(self.root))

    def create(self) -> Path:
        path = self.directory.data / "out.txt"
        path.write_text(self.name, encoding="utf-8")
        return path


class Failing(Spec):
    name: str

    def create(self) -> str:
        (self.directory.data / "partial.txt").write_text("half", encoding="utf-8")
        raise RuntimeError(f"failing {self.name}")


class Row(Spec):
    index: int

    @batched(lambda _, __: (None, 8))
    def create(objs: list["Row"], directory: furu.SpecDirectory) -> list[str]:
        return [f"{_says[0]} row {obj.index}" for obj in objs]


class Sweep(Spec):
    size: int

    @furu.dependency(remember=True)
    def parts(self) -> list[Corpus]:
        _calls.append(self.size)
        return [Corpus(name=f"part{index}") for index in range(self.size)]

    def create(self) -> int:
        return len(furu.create(self.parts))


@contextmanager
def overlaid(folder: Path) -> Generator[None]:
    """A session started with ``FURU_OVERLAY=folder`` in its environment."""
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("FURU_OVERLAY", str(folder))
        config = get_config().model_copy(update={"overlay": _Config().overlay})
        with override_config(config):
            yield


def _tree(*roots: Path) -> dict[str, tuple[int, bytes | None]]:
    """Every folder and file under ``roots``: when it last changed, and what it holds."""
    return {
        str(path): (
            path.lstat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for root in roots
        for path in (root, *root.rglob("*"))
        if path.exists()
    }


def _real(*more: Path) -> dict[str, tuple[int, bytes | None]]:
    """The real store as it is now: every data directory, and ``more``."""
    return _tree(*(path for _, path in get_config().directories), *more)


@pytest.fixture
def overlay(tmp_path: Path) -> Path:
    return tmp_path / "overlay"


@pytest.fixture(autouse=True)
def _a_fresh_process() -> None:
    _says[0] = "real"
    _calls.clear()
    fresh_process()


# --- configuration ------------------------------------------------------------


def test_the_environment_sets_the_overlay(
    monkeypatch: pytest.MonkeyPatch, overlay: Path
) -> None:
    assert _Config().overlay is None

    monkeypatch.setenv("FURU_OVERLAY", str(overlay))

    assert _Config().overlay == overlay
    assert not overlay.exists()  # claimed when first used


def test_an_empty_variable_is_no_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURU_OVERLAY", "")

    assert _Config().overlay is None


def test_pyproject_cannot_set_the_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.furu]\noverlay = "/tmp/furu-overlay"\n')

    with pytest.raises(ValueError, match="pyproject.toml sets a furu overlay"):
        _Config()

    # The environment still may, beside a pyproject that sets something else.
    pyproject.write_text('tool.furu.code_version = "traced"\n')
    monkeypatch.setenv("FURU_OVERLAY", str(tmp_path / "overlay"))
    config = _Config()
    assert (config.code_version, config.overlay) == ("traced", tmp_path / "overlay")


def test_the_overlay_must_be_an_absolute_path() -> None:
    with (
        overlaid(Path("relative/overlay")),
        pytest.raises(ValueError, match="must be an absolute path"),
    ):
        Corpus(name="a").create()

    assert not Path("relative").exists()


def test_pytest_does_not_inherit_the_shells_overlay(
    tmp_path: Path, overlay: Path
) -> None:
    test = tmp_path / "test_inner.py"
    test.write_text(
        "import os\n"
        "import furu\n"
        "from furu.config import get_config\n\n"
        "class Inner(furu.Spec):\n"
        "    def create(self) -> int:\n"
        "        return 1\n\n"
        "def test_inner():\n"
        "    assert 'FURU_OVERLAY' not in os.environ\n"
        "    assert get_config().overlay is None\n"
        "    assert not Inner().done\n",
        encoding="utf-8",
    )

    for shell in (str(overlay), "relative/overlay"):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test)],
            cwd=tmp_path,
            env=os.environ | {"FURU_OVERLAY": shell},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert not overlay.exists()
    assert not (tmp_path / "relative").exists()


# --- claiming the folder ------------------------------------------------------


def test_a_new_folder_is_claimed_and_claimed_again_by_the_next_session(
    overlay: Path,
) -> None:
    with overlaid(overlay / "nested"):
        assert Corpus(name="a").create() == "corpus a"
    mark = overlay / "nested" / ".gitignore"
    assert mark.read_text().endswith("\n*\n")
    claimed = mark.stat().st_mtime_ns

    _claim.cache_clear()  # as the next process
    with overlaid(overlay / "nested"):
        assert Corpus(name="a").done
        assert Corpus(name="b").create() == "corpus b"
    assert mark.stat().st_mtime_ns == claimed


def test_an_empty_folder_is_claimed(overlay: Path) -> None:
    overlay.mkdir()

    with overlaid(overlay):
        assert Corpus(name="a").create() == "corpus a"

    assert (overlay / ".gitignore").is_file()


@pytest.mark.parametrize("existing", ["folder", "file"])
def test_an_existing_folder_or_file_is_never_claimed(
    overlay: Path, existing: str
) -> None:
    if existing == "folder":
        overlay.mkdir()
        (overlay / "thesis.tex").write_text("years of work", encoding="utf-8")
        (overlay / ".gitignore").write_text("*.aux\n", encoding="utf-8")
    else:
        overlay.write_text("years of work", encoding="utf-8")
    before = _tree(overlay)

    with overlaid(overlay):
        for use in (
            lambda: Corpus(name="a").create(),
            lambda: Corpus(name="a").done,
            lambda: get_config().directories.anchored(),
            lambda: get_config().manager_version_dir,
        ):
            with pytest.raises(RuntimeError, match="is not a furu overlay"):
                use()

    assert _tree(overlay) == before
    assert [path.name for path in overlay.parent.iterdir()] == [overlay.name]
    assert not Corpus(name="a").done


def test_an_overlay_inside_a_git_repository_stays_out_of_git_status(
    git_repo: Path,
) -> None:
    overlay = git_repo / "scratch" / "overlay"

    with overlaid(overlay):
        assert Corpus(name="a").create() == "corpus a"
        assert Elsewhere(root=str(git_repo / "results"), name="x").create().is_file()
        # What keeps furu's data directories out of git status goes there too.
        require_pushed(GitIdentity.capture(git_repo))

    assert any(overlay.rglob("out.txt"))
    assert git(git_repo, "status", "--porcelain", "--untracked-files=all") == ""
    assert sorted(path.name for path in git_repo.iterdir()) == [
        ".git",
        "scratch",
        "tracked.txt",
    ]


# --- reading the real store, writing the overlay ------------------------------


def test_paths_are_mirrored_under_the_overlay(overlay: Path, tmp_path: Path) -> None:
    real = get_config().directories.anchored()
    path = tmp_path / "store" / "file"
    assert (upper(path), lower(path), layers(path)) == (path, path, [path])

    with overlaid(overlay):
        mirrored = overlay / path.relative_to("/")
        assert upper(path) == upper(mirrored) == mirrored
        assert lower(path) == lower(mirrored) == path
        assert layers(path) == layers(mirrored) == [mirrored, path]
        # Plans, records, snapshots, logs and the manager's files are written.
        written = get_config().directories.anchored()
        assert written == type(real)(**{name: upper(folder) for name, folder in real})
        assert all(folder.is_relative_to(overlay) for _, folder in written)
        assert get_config().manager_version_dir.is_relative_to(overlay)
        assert Corpus(name="a")._class_dir.is_relative_to(written.objects)

    assert get_config().directories.anchored() == real


def test_a_session_reads_the_real_store_and_writes_only_to_its_overlay(
    overlay: Path,
) -> None:
    assert Corpus(name="a").create() == "corpus a"
    stored = run_of(Corpus(name="a"))
    before = _real()

    with overlaid(overlay):
        corpus = Corpus(name="a")
        assert corpus.done
        assert run_of(corpus) == stored  # no copy: read where it is
        assert corpus.load_existing() == "corpus a"
        tokens = Tokens(corpus=corpus)
        assert not tokens.done
        assert tokens.create() == f"CORPUS A|{os.getpid()}"
        assert tokens.done
        assert run_of(tokens).is_relative_to(overlay)
        assert furu.load_existing([corpus, tokens]) == [
            "corpus a",
            f"CORPUS A|{os.getpid()}",
        ]
        # The plan, the record and the schema file are the session's too.
        written = get_config().directories.anchored()
        assert list(written.plans.glob("*.plan"))
        assert list(written.provenance.glob("launch-*"))
        assert schema_path_in(tokens._class_dir).is_file()
        assert not attempt_parent([tokens]).exists()

    assert _real() == before
    assert not Tokens(corpus=Corpus(name="a")).done

    # The next session with the overlay goes on from it, and deleting the
    # folder undoes both.
    with overlaid(overlay):
        assert Tokens(corpus=Corpus(name="a")).done
    shutil.rmtree(overlay)
    assert _real() == before
    assert Corpus(name="a").load_existing() == "corpus a"


def test_the_overlays_result_is_read_before_the_real_stores(overlay: Path) -> None:
    _says[0] = "overlay"
    with overlaid(overlay):
        assert Echo(name="a").create() == "overlay"
    _says[0] = "real"
    assert not Echo(name="a").done
    assert Echo(name="a").create() == "real"

    with overlaid(overlay):
        echo = Echo(name="a")
        assert echo.create() == "overlay"
        assert run_of(echo).is_relative_to(overlay)
    assert Echo(name="a").create() == "real"


def test_a_working_folder_in_the_real_store_is_neither_taken_nor_waited_for(
    overlay: Path,
) -> None:
    with pytest.raises(RuntimeError, match="failing a"):
        Failing(name="a").create()
    held = attempt_parent([Corpus(name="b")]) / "local__elsewhere__1__9f3c2a7d51b0e6a4"
    held.mkdir(parents=True)
    before = _real()

    with overlaid(overlay):
        with pytest.raises(RuntimeError, match="failing a"):
            Failing(name="a").create()
        assert attempt_parent([Failing(name="a")]).is_relative_to(overlay)
        assert Corpus(name="b").create() == "corpus b"

    assert _real() == before


def test_custom_storage_is_mirrored_under_the_overlay(
    overlay: Path, tmp_path: Path
) -> None:
    custom = tmp_path / "custom"
    stored = Elsewhere(root=str(custom), name="kept").create()
    assert stored.is_relative_to(custom)
    before = _real(custom)

    with overlaid(overlay):
        assert Elsewhere(root=str(custom), name="kept").create() == stored
        new = Elsewhere(root=str(custom), name="new").create()
        assert new.is_relative_to(upper(custom))
        assert new.read_text() == "new"
        untouched = tmp_path / "untouched"
        assert Elsewhere(root=str(untouched), name="x").create().read_text() == "x"
        assert not untouched.exists()

    assert _real(custom) == before
    assert not Elsewhere(root=str(custom), name="new").done


def test_a_batch_runs_only_the_specs_that_neither_store_holds(overlay: Path) -> None:
    assert furu.create([Row(index=0), Row(index=1)]) == ["real row 0", "real row 1"]
    before = _real()

    _says[0] = "overlay"
    with overlaid(overlay):
        rows = [Row(index=index) for index in range(3)]
        assert furu.create(rows) == ["real row 0", "real row 1", "overlay row 2"]
        assert run_of(rows[2]).is_relative_to(overlay)
        assert not run_of(rows[0]).is_relative_to(overlay)
        with pytest.raises(RuntimeError, match="@furu.batched has no folder"):
            rows[2].delete(mode="force")

    assert _real() == before


# --- delete -------------------------------------------------------------------


def test_delete_removes_what_the_overlay_holds(overlay: Path) -> None:
    before = _real()

    with overlaid(overlay):
        corpus, failing = Corpus(name="a"), Failing(name="a")
        assert corpus.create() == "corpus a"
        with pytest.raises(RuntimeError, match="failing a"):
            failing.create()
        run, working = run_of(corpus), attempt_parent([failing])
        assert working.is_relative_to(overlay)
        assert any(working.rglob("partial.txt"))

        with patch("builtins.input", return_value="n"):
            assert not corpus.delete()
        assert corpus.delete(mode="force")
        assert failing.delete(mode="force")

        assert not run.exists()
        assert not working.exists()
        assert not corpus.done
        assert not corpus.delete(mode="force")
        assert corpus.create() == "corpus a"

    assert _real() == before


def test_delete_refuses_a_result_in_the_real_store_and_touches_nothing(
    overlay: Path,
) -> None:
    assert Corpus(name="a").create() == "corpus a"
    before = _real()

    with overlaid(overlay):
        corpus = Corpus(name="a")
        with (
            patch("builtins.input", side_effect=AssertionError("asked")),
            pytest.raises(RuntimeError, match="is in the real store") as refused,
        ):
            corpus.delete()
        assert str(lower(run_of(corpus))) in str(refused.value)
        with pytest.raises(RuntimeError, match="Nothing was deleted"):
            corpus.delete(mode="force")
        assert corpus.done
        assert corpus.load_existing() == "corpus a"

    assert _real() == before
    # Nothing but the claim: no folder for the class, and no working folder.
    assert [path.name for path in overlay.rglob("*")] == [".gitignore"]


def test_delete_refuses_while_both_stores_hold_the_result(overlay: Path) -> None:
    _says[0] = "overlay"
    with overlaid(overlay):
        assert Echo(name="a").create() == "overlay"
        with pytest.raises(RuntimeError, match="failing a"):
            Failing(name="a").create()
    _says[0] = "real"
    assert Echo(name="a").create() == "real"
    before, held = _real(), _tree(overlay)

    with overlaid(overlay):
        with pytest.raises(RuntimeError, match="is in the real store"):
            Echo(name="a").delete(mode="force")
        assert Echo(name="a").load_existing() == "overlay"
        assert _tree(overlay) == held
        # What only the overlay holds is still deleted.
        assert Failing(name="a").delete(mode="force")

    assert _real() == before


# --- code_version="traced" ----------------------------------------------------

BASE = """\
import furu


def scale(n):
    return n * 2


class Base(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return scale(self.n)
"""

TOP = """\
import furu

from PKG.base import Base


def shift(n):
    return n + 1


class Top(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return shift(Base(n=self.n).create())
"""


@pytest.fixture
def traced(repo: Repo) -> ModuleType:
    repo.write("base.py", BASE)
    repo.write("top.py", TOP)
    return repo.load("top")


def _version(spec: Spec) -> Path:
    return run_of(spec).parent.parent


def test_traced_results_are_live_across_both_stores(
    repo: Repo, traced: ModuleType, overlay: Path
) -> None:
    assert traced.Base(n=1).create() == 2
    real_version = _version(traced.Base(n=1))
    before = _real()

    with overlaid(overlay):
        fresh_process()
        base, top = traced.Base(n=1), traced.Top(n=1)
        assert base.done
        assert top.create() == 3
        assert _version(base) == real_version
        assert _version(top).is_relative_to(overlay)
        # The trace names the code version it was built on as it is named
        # without an overlay, whichever store holds it.
        trace = json.loads(trace_path_in(_version(top)).read_text())
        assert trace["built_on"] == [
            os.path.relpath(real_version, lower(get_config().directories.objects))
        ]
        first = _version(top)
        fresh_process()
        assert traced.Top(n=1).done

        # Code that changed leaves the real store's result behind, and what
        # was built on it: both run again, in the overlay.
        repo.edit("base.py", "n * 2", "n * 3")
        traced = repo.load("top")
        base, top = traced.Base(n=1), traced.Top(n=1)
        assert not base.done
        assert not top.done
        assert top.create() == 4
        assert _version(base).is_relative_to(overlay)
        assert _version(base).name != real_version.name
        assert _version(top) != first
        fresh_process()
        assert traced.Top(n=1).done

        # The real store's run no longer counts, so it does not stop a delete.
        assert traced.Base(n=1).delete(mode="force")
        assert not traced.Base(n=1).done
        assert real_version.is_dir()

    assert _real() == before
    assert not traced.Base(n=1).done
    assert not traced.Top(n=1).done

    # Reverting makes the real store's version live again, and with it the
    # overlay's first result, which was built on it.
    repo.edit("base.py", "n * 3", "n * 2")
    traced = repo.load("top")
    assert traced.Base(n=1).done
    assert not traced.Top(n=1).done
    with overlaid(overlay):
        top = traced.Top(n=1)
        assert top.done
        assert _version(top) == first
        assert top.load_existing() == 3
    assert _real() == before


def test_a_code_version_in_both_stores_keeps_its_runs_apart(
    traced: ModuleType, overlay: Path
) -> None:
    assert traced.Base(n=1).create() == 2
    before = _real()

    with overlaid(overlay):
        fresh_process()
        assert traced.Base(n=2).create() == 4
        mine, theirs = _version(traced.Base(n=2)), _version(traced.Base(n=1))
        assert mine.is_relative_to(overlay)
        assert (mine.name, lower(mine)) == (theirs.name, theirs)
        assert trace_path_in(mine).read_bytes() == trace_path_in(theirs).read_bytes()
        fresh_process()
        assert traced.Base(n=1).done
        assert traced.Base(n=2).done

    assert _real() == before
    assert not traced.Base(n=2).done


# --- schema, migrations and what dependency methods returned ------------------


def _merge(*, migrations: object = (), **fields: Any) -> Any:
    """The spec class Merge as its source reads at one point in time.

    A field is ``name=type`` or ``name=(type, default)``.
    """
    body: dict[str, Any] = {
        "__module__": __name__,
        "__annotations__": {},
        "migrations": migrations,
        "create": lambda self: _says[0],
    }
    for name, declared in fields.items():
        if isinstance(declared, tuple):
            body["__annotations__"][name], body[name] = declared
        else:
            body["__annotations__"][name] = declared
    return types.new_class("Merge", (Spec,), exec_body=lambda ns: ns.update(body))


def _schema_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_a_declared_default_is_recorded_in_the_overlay_and_compared_with_both(
    overlay: Path,
) -> None:
    before_class = _merge(lemma=str)
    assert before_class(lemma="comer").create() == "real"
    real_schema = schema_path_in(before_class(lemma="comer")._class_dir)
    before = _real()

    def added(default: float) -> Any:
        return _merge(
            lemma=str,
            min_overlap=(float, default),
            migrations=(Added("min_overlap", default=default),),
        )

    _says[0] = "overlay"
    with overlaid(overlay):
        # At its default the spec is the one stored before the field existed.
        assert added(0.5)(lemma="comer").create() == "real"
        schema = schema_path_in(added(0.5)(lemma="comer")._class_dir)
        assert schema == upper(real_schema)
        assert _schema_lines(schema) == [{"added": "min_overlap", "default": 0.5}]
        with pytest.raises(MigrationError, match="recorded 0.5"):
            _ = added(0.8)(lemma="comer").done

        other = added(0.5)(lemma="comer", min_overlap=0.8)
        assert other.create() == "overlay"
        assert run_of(other).is_relative_to(overlay)
        assert _schema_lines(schema) == [{"added": "min_overlap", "default": 0.5}]

    assert _real() == before
    assert _schema_lines(real_schema) == [{"names": ["lemma"]}]
    # The real store never recorded the default, so there it can still be chosen.
    assert added(0.8)(lemma="comer").done

    # What the real store records comes first, also for an overlay made earlier.
    assert _schema_lines(real_schema)[1:] == [{"added": "min_overlap", "default": 0.8}]
    with overlaid(overlay), pytest.raises(MigrationError, match="recorded 0.8"):
        _ = added(0.5)(lemma="comer").done


def test_a_launch_asks_about_unexplained_results_in_both_stores(
    overlay: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _merge(lemma=str)(lemma="comer")
    assert old.create() == "real"
    before = _real()

    _says[0] = "overlay"
    with overlaid(overlay):
        changed = _merge(lemma=str, window=(int, 3))(lemma="comer")
        with pytest.raises(MigrationError, match="leave stored results behind") as asks:
            changed.create()
        assert f"{lower(old._class_dir)} was stored without" in str(asks.value)
        assert f"stores in {changed._class_dir}" in str(asks.value)
        assert not changed._class_dir.exists()  # asking stores nothing

        ignore = f"{__name__}.Merge:{old._class_part}"
        monkeypatch.setenv("FURU_IGNORE_STORED", ignore)
        assert changed.create() == "overlay"
        monkeypatch.delenv("FURU_IGNORE_STORED")

        # What the session itself left behind is asked about as well.
        again = _merge(lemma=str, stride=(int, 1))(lemma="comer")
        with pytest.raises(MigrationError) as asks:
            again.create()
        assert f"{lower(old._class_dir)} was stored" in str(asks.value)
        assert f"{changed._class_dir} was stored" in str(asks.value)

    assert _real() == before


def test_a_remembered_dependency_is_found_in_either_store(overlay: Path) -> None:
    assert Sweep(size=2).create() == 2
    assert _calls == [2]
    _calls.clear()
    fresh_process()
    before = _real()

    with overlaid(overlay):
        # Remembered by the real store: not called, and its result is there.
        assert Sweep(size=2).create() == 2
        assert _calls == []

        new = Sweep(size=3)
        assert new.create() == 3
        assert _calls == [3]
        remembered = remembered_dir_in(new._class_dir)
        assert remembered.is_relative_to(overlay)
        assert len(list(remembered.glob("launch-*.jsonl"))) == 1
        assert list((remembered / "traces").iterdir())

        _calls.clear()
        fresh_process()
        assert Sweep(size=3).done
        assert Sweep(size=2).done
        assert _calls == []

    assert _real() == before
    fresh_process()
    assert not Sweep(size=3).done
    assert _calls == [3]


# --- workers ------------------------------------------------------------------


def test_workers_read_the_real_store_and_write_to_the_overlay(
    overlay: Path, tmp_path: Path
) -> None:
    assert Corpus(name="a").create() == "corpus a"
    custom = tmp_path / "custom"
    before = _real(custom)

    with overlaid(overlay):
        tokens = Tokens(corpus=Corpus(name="a"))
        elsewhere = Elsewhere(root=str(custom), name="x")
        made, path = furu.create(
            (tokens, elsewhere), on=(LocalThreadWorkerBackend(max_workers=2),)
        )

        text, _, pid = made.partition("|")
        assert text == "CORPUS A"
        assert int(pid) != os.getpid()  # a worker's child process ran it
        assert path.read_text() == "x"
        assert path.is_relative_to(upper(custom))
        assert run_of(tokens).is_relative_to(overlay)
        assert not run_of(Corpus(name="a")).is_relative_to(overlay)
        written = get_config().directories.anchored()
        assert list(written.plans.glob("*.plan"))
        assert list(written.executions.glob("*/execution.log"))
        # The launch's file, and one of each child that ran a create().
        assert len(list(written.provenance.iterdir())) == 3

        # A second launch finds everything, and a worker's child finds what
        # the first one's children wrote.
        again = Tokens(corpus=Corpus(name="b"))
        assert furu.create(again, on=(LocalThreadWorkerBackend(),)).startswith(
            "CORPUS B|"
        )
        assert run_of(Corpus(name="b")).is_relative_to(overlay)

    assert _real(custom) == before
    assert not custom.exists()
    assert not Tokens(corpus=Corpus(name="a")).done
    assert not Corpus(name="b").done
