"""An overlay: a debug session reads what is stored and writes beside it."""

import shutil
from pathlib import Path

import pytest
from runs import run_of, stored_files, working_folder

import furu
from furu import Metadata, Spec
from furu.config import _Config, get_config
from furu.storage.attempt_lock import HeldError
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend

# The name of each Corpus this process created.
_created: list[str] = []


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
    def create(specs: list["Pair"], directory: furu.SpecDirectory) -> list[int]:
        return [spec.n for spec in specs]


@pytest.fixture(autouse=True)
def _nothing_created() -> None:
    _created.clear()


def _in(overlay: Path, directory: Path) -> Path:
    return overlay / directory.relative_to(directory.anchor)


def test_a_session_reads_what_is_stored_and_writes_only_to_the_overlay(
    overlaid: _Config,
) -> None:
    assert Corpus(name="a").create() == "A"
    stored = stored_files()
    directories = get_config().directories
    assert overlaid.overlay is not None

    with override_config(overlaid):
        corpus, tokens = Corpus(name="a"), Tokens(corpus=Corpus(name="a"))
        path = tokens.create()

        assert path.read_text() == "A"
        assert _created == ["a"]  # the stored corpus was loaded
        assert run_of(corpus).is_relative_to(directories.objects)
        assert run_of(tokens).is_relative_to(_in(overlaid.overlay, directories.objects))
        assert path.is_relative_to(run_of(tokens))
        assert corpus.done and tokens.done

    assert stored_files() == stored
    assert not Tokens(corpus=Corpus(name="a")).done
    # The launch's plan and record are in the overlay as well.
    for written in (directories.plans, directories.provenance):
        assert list(_in(overlaid.overlay, written).iterdir())

    shutil.rmtree(overlaid.overlay)
    with override_config(overlaid):
        assert Corpus(name="a").done
        assert not Tokens(corpus=Corpus(name="a")).done


def test_workers_read_and_write_as_the_session_that_started_them(
    overlaid: _Config,
) -> None:
    Corpus(name="a").create()
    stored = stored_files()
    assert overlaid.overlay is not None

    with override_config(overlaid):
        tokens = Tokens(corpus=Corpus(name="a"))
        path = furu.create(tokens, on=[LocalThreadWorkerBackend()])

        assert path.read_text() == "A"
        assert path.is_relative_to(overlaid.overlay)
        executions = _in(overlaid.overlay, get_config().directories.executions)
        assert list(executions.glob("*/execution.log"))

    assert stored_files() == stored


def test_storage_a_class_names_is_mirrored_in_the_overlay(
    overlaid: _Config, tmp_path: Path
) -> None:
    root = tmp_path / "kept"
    assert Kept(root=str(root), n=1).create() == 1
    kept = sorted(root.rglob("*"))
    assert overlaid.overlay is not None

    with override_config(overlaid):
        assert run_of(Kept(root=str(root), n=1)).is_relative_to(root)
        new = Kept(root=str(root), n=2)
        assert new.create() == 2
        assert run_of(new).is_relative_to(_in(overlaid.overlay, root))

    assert sorted(root.rglob("*")) == kept


def test_a_batch_in_the_overlay_holds_only_the_specs_not_stored_under_it(
    overlaid: _Config,
) -> None:
    assert furu.create([Pair(n=1), Pair(n=2)]) == [1, 2]
    stored = stored_files()
    assert overlaid.overlay is not None

    with override_config(overlaid):
        assert furu.create([Pair(n=1), Pair(n=2), Pair(n=3)]) == [1, 2, 3]
        assert run_of(Pair(n=1)) == run_of(Pair(n=2))
        assert not run_of(Pair(n=1)).is_relative_to(overlaid.overlay)
        assert run_of(Pair(n=3)).is_relative_to(overlaid.overlay)

    assert stored_files() == stored


def test_delete_removes_what_is_in_the_overlay_and_refuses_what_is_under_it(
    overlaid: _Config,
) -> None:
    Corpus(name="a").create()
    stored = stored_files()

    with override_config(overlaid):
        with pytest.raises(RuntimeError, match="under the overlay, which only reads"):
            Corpus(name="a").delete(mode="force")
        new = Corpus(name="b")
        new.create()
        assert new.delete(mode="force")
        assert not new.done

    assert stored_files() == stored


def test_a_working_folder_under_the_overlay_holds_nobody_up(overlaid: _Config) -> None:
    with pytest.raises(KeyboardInterrupt):
        Stops().create()  # killed with files in its working folder
    folder = working_folder(Stops())
    folder.rename(folder.with_name("local__elsewhere__1__token"))
    with pytest.raises(HeldError, match="process 1 on elsewhere"):
        Stops().create()
    stored = stored_files()

    with override_config(overlaid):
        assert Stops().create() == 1

    assert stored_files() == stored


def test_an_overlay_is_an_absolute_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURU_OVERLAY", "/tmp/furu-debug")
    assert _Config().overlay == Path("/tmp/furu-debug")
    assert _Config().anchored().overlay == Path("/tmp/furu-debug")

    monkeypatch.setenv("FURU_OVERLAY", "furu-debug")
    with pytest.raises(ValueError, match="overlay must be an absolute path"):
        _Config()
