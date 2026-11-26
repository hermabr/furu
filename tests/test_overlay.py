"""An overlay: a debug session reads what is stored and writes beside it."""

import itertools
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from runs import read_only, run_of, stored_files, working_folder

import furu
import furu.execution.run_on_workers as run_on_workers_module
from furu import Metadata, Spec, testing
from furu.config import _Config, get_config
from furu.execution.manager import Manager
from furu.storage.attempt_lock import HeldError
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend

_serve = Manager.serve
# The name of each Corpus this process created.
_created: list[str] = []
_runs = itertools.count()


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


class Counted(Spec):
    """Each create() returns a new number, so two results of one spec are told apart."""

    name: str

    def create(self) -> int:
        return next(_runs)


class Gated(Spec):
    """Notes each create() call, then waits until the test opens the gate."""

    gate: str

    def create(self) -> int:
        gate = Path(self.gate)
        (gate / f"started-{os.getpid()}-{time.monotonic_ns()}").touch()
        while not (gate / "open").exists():
            time.sleep(0.02)
        return 1


def _starts(gate: Path) -> int:
    return len(list(gate.glob("started-*")))


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


def test_a_session_reads_what_is_stored_and_writes_only_to_the_overlay(
    session: _Config, overlay: Path
) -> None:
    assert Corpus(name="a").create() == "A"
    stored = stored_files()
    directories = get_config().directories

    # Nothing under the overlay can be written, so nothing there is locked either.
    with read_only(), override_config(session):
        corpus, tokens = Corpus(name="a"), Tokens(corpus=Corpus(name="a"))
        path = tokens.create()

        assert path.read_text() == "A"
        assert _created == ["a"]  # the stored corpus was loaded
        assert run_of(corpus).is_relative_to(directories.objects)
        assert run_of(tokens).is_relative_to(session.overlaid(directories.objects))
        assert path.is_relative_to(run_of(tokens))
        assert corpus.done and tokens.done
        # A corpus that is not stored is created in the overlay.
        assert Corpus(name="b").create() == "B"
        assert run_of(Corpus(name="b")).is_relative_to(overlay)

    assert stored_files() == stored
    assert not Tokens(corpus=Corpus(name="a")).done
    # The launch's plan and record are in the overlay as well.
    for written in (directories.plans, directories.provenance):
        assert list(session.overlaid(written).iterdir())

    shutil.rmtree(overlay)
    with override_config(session):
        assert Corpus(name="a").done
        assert not Tokens(corpus=Corpus(name="a")).done


def test_workers_read_and_write_as_the_session_that_started_them(
    session: _Config, overlay: Path
) -> None:
    Corpus(name="a").create()
    stored = stored_files()

    with read_only(), override_config(session):
        tokens = Tokens(corpus=Corpus(name="a"))
        path = furu.create(tokens, on=[LocalThreadWorkerBackend()])

        assert path.read_text() == "A"
        assert path.is_relative_to(overlay)
        executions = session.overlaid(get_config().directories.executions)
        assert list(executions.glob("*/execution.log"))

    assert stored_files() == stored


def test_a_result_in_the_overlay_is_the_one_that_counts(
    session: _Config, overlay: Path, tmp_path: Path
) -> None:
    assert Counted(name="a").create() == 0
    # Stored first, and not there yet when the session creates the spec too.
    class_dir = Counted(name="a")._class_dir
    shutil.move(class_dir, tmp_path / "aside")
    with override_config(session):
        assert Counted(name="a").create() == 1
    shutil.move(tmp_path / "aside", class_dir)

    with override_config(session):
        spec = Counted(name="a")
        assert spec.load_existing() == 1
        assert run_of(spec).is_relative_to(overlay)
    assert Counted(name="a").load_existing() == 0

    shutil.rmtree(overlay)
    with override_config(session):
        assert Counted(name="a").load_existing() == 0


def test_storage_a_class_names_is_mirrored_in_the_overlay(
    session: _Config, tmp_path: Path
) -> None:
    root = tmp_path / "kept"
    assert Kept(root=str(root), n=1).create() == 1
    kept = sorted(root.rglob("*"))

    with override_config(session):
        assert run_of(Kept(root=str(root), n=1)).is_relative_to(root)
        new = Kept(root=str(root), n=2)
        assert new.create() == 2
        assert run_of(new).is_relative_to(session.overlaid(root))

    assert sorted(root.rglob("*")) == kept


def test_a_batch_in_the_overlay_holds_only_the_specs_not_stored_under_it(
    session: _Config, overlay: Path
) -> None:
    assert furu.create([Pair(n=1), Pair(n=2)]) == [1, 2]
    stored = stored_files()

    with read_only(), override_config(session):
        assert furu.create([Pair(n=1), Pair(n=2), Pair(n=3)]) == [1, 2, 3]
        assert run_of(Pair(n=1)) == run_of(Pair(n=2))
        assert not run_of(Pair(n=1)).is_relative_to(overlay)
        assert run_of(Pair(n=3)).is_relative_to(overlay)

    assert stored_files() == stored


def test_delete_removes_what_is_in_the_overlay_and_refuses_what_is_under_it(
    session: _Config,
) -> None:
    Corpus(name="a").create()
    stored = stored_files()

    with override_config(session):
        with pytest.raises(RuntimeError, match="under the overlay, which only reads"):
            Corpus(name="a").delete(mode="force")
        new = Corpus(name="b")
        new.create()
        assert new.delete(mode="force")
        assert not new.done

    assert stored_files() == stored


def test_a_working_folder_under_the_overlay_holds_nobody_up(session: _Config) -> None:
    with pytest.raises(KeyboardInterrupt):
        Stops().create()  # killed with files in its working folder
    folder = working_folder(Stops())
    folder.rename(folder.with_name("local__elsewhere__1__token"))
    with pytest.raises(HeldError, match="process 1 on elsewhere"):
        Stops().create()
    stored = stored_files()

    with read_only(), override_config(session):
        assert Stops().create() == 1

    assert stored_files() == stored


def _wait_until(condition: Callable[[], bool], *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        time.sleep(0.02)


@contextmanager
def _manager() -> Generator[Manager]:
    """A shared manager that scripts find as the config says, as Slurm runs do."""
    with _serve(
        listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
    ) as manager:
        address = get_config().manager_version_dir / "address.json"
        address.parent.mkdir(parents=True)
        address.write_text(json.dumps({"url": manager.url}))
        yield manager


def test_a_session_and_the_store_each_run_a_spec_under_a_manager_of_their_own(
    session: _Config,
    overlay: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Scripts with local backends use the shared manager, not their own.
    monkeypatch.setattr(Manager, "serve", lambda **_: nullcontext(None))
    monkeypatch.setattr(run_on_workers_module, "_MANAGER_RETRY_INTERVAL_S", 0.1)
    stores = get_config().model_copy(update={"manager_dir": tmp_path / "manager"})
    session = session.model_copy(update={"manager_dir": tmp_path / "manager"})
    assert session.manager_version_dir.is_relative_to(overlay)
    gate = tmp_path / "gate"
    gate.mkdir()
    outcomes: dict[str, object] = {}

    @contextmanager
    def creating(name: str) -> Generator[Gated]:
        """A launch of the one spec, under the config there is now."""
        spec = Gated(gate=str(gate))
        launch = threading.Thread(
            target=lambda: outcomes.update(
                {name: furu.create(spec, on=[LocalThreadWorkerBackend()])}
            )
        )
        launch.start()
        try:
            yield spec
        finally:
            (gate / "open").touch()
            launch.join(timeout=30)
            assert not launch.is_alive()

    with override_config(stores), _manager() as of_store, creating("store") as stored:
        _wait_until(lambda: _starts(gate) == 1)
        with override_config(session), creating("session") as debugged:
            assert debugged.object_id == stored.object_id
            # The store's manager is not the session's: it knows a spec by its
            # id alone, and would take this one for the one it is running.
            waits = f"start one with FURU_OVERLAY={overlay} uvx furu@"
            _wait_until(lambda: waits in caplog.text)
            with of_store.lock:
                assert len(of_store.scripts) == 1
            with _manager() as of_session:
                _wait_until(lambda: _starts(gate) == 2)
                with of_store.lock, of_session.lock:
                    assert len(of_store.leases) == len(of_session.leases) == 1
                (gate / "open").touch()
                _wait_until(lambda: "session" in outcomes)
            assert run_of(debugged).is_relative_to(overlay)
        _wait_until(lambda: "store" in outcomes)
        assert not run_of(stored).is_relative_to(overlay)
    assert outcomes == {"store": 1, "session": 1}


@pytest.mark.parametrize(
    ("value", "overlay"),
    [
        ("/tmp/furu-debug", Path("/tmp/furu-debug")),
        ("/tmp/x/../furu-debug/", Path("/tmp/furu-debug")),
        ("", None),
    ],
)
def test_an_overlay_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, overlay: Path | None
) -> None:
    monkeypatch.setenv("FURU_OVERLAY", value)
    assert _Config().overlay == overlay
    assert _Config().anchored().overlay == overlay
    # As a worker's process reads the config it is sent.
    assert _Config.model_validate_json(_Config().model_dump_json()).overlay == overlay


@pytest.mark.parametrize("value", ["furu-debug", ".", "/", "/tmp/.."])
def test_an_overlay_is_an_absolute_path_to_a_folder_of_its_own(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("FURU_OVERLAY", value)
    with pytest.raises(ValueError, match="overlay must be an absolute path"):
        _Config()


def test_an_overlay_and_what_it_stands_for_do_not_hold_one_another(
    tmp_path: Path,
) -> None:
    objects = get_config().directories.anchored().objects
    for overlay in (
        objects,
        objects / "debug",
        objects.parent,
        objects / ".." / objects.name,
    ):
        session = _Config.model_validate(
            get_config().model_dump() | {"overlay": overlay}
        )
        with (
            override_config(session),
            pytest.raises(ValueError, match="must not hold one another"),
        ):
            Corpus(name="a").create()
        with pytest.raises(ValueError, match="must not hold one another"):
            _ = session.write_directories

    # Storage that a class names, too.
    session = get_config().model_copy(update={"overlay": tmp_path / "overlay"})
    with (
        override_config(session),
        pytest.raises(ValueError, match="must not hold one another"),
    ):
        Kept(root=str(tmp_path / "overlay" / "kept"), n=1).create()
    assert not (tmp_path / "overlay").exists()


def test_tests_run_without_an_overlay(session: _Config) -> None:
    with override_config(session):
        testing.pytest_configure(SimpleNamespace(stash={}))  # ty: ignore[invalid-argument-type]
        assert get_config().overlay is None
