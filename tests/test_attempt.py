"""A run's working folder, held by its name."""

import os
import socket
import subprocess
from multiprocessing import get_context
from pathlib import Path

import pytest

from furu.locking import short_lock
from furu.storage import attempt
from furu.storage.attempt import Held, claimed, current, why_held

HOST = socket.gethostname()


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attempt, "WAIT_S", 0.0)


@pytest.fixture
def parent(tmp_path: Path) -> Path:
    return tmp_path / "attempts" / "03be"


def _dead_pid() -> int:
    child = get_context("fork").Process(target=lambda: None)
    child.start()
    child.join()
    assert child.pid is not None
    return child.pid


def _folder(parent: Path, name: str) -> Path:
    (parent / name).mkdir(parents=True)
    (parent / name / "log").write_text("earlier work")
    return parent / name


def _slurm(monkeypatch: pytest.MonkeyPatch, *, queued: str = "", states: str) -> None:
    def answer(*argv: str) -> str:
        assert argv[0] in ("squeue", "sacct")
        if argv[0] == "sacct":
            assert argv[1:] == (
                *["-D", "-X", "--noheader", "--parsable2", "-o", "State", "-j"],
                "7",
            )
        return queued if argv[0] == "squeue" else states

    monkeypatch.setattr(attempt, "_slurm", answer)
    monkeypatch.setattr(attempt, "_cluster_name", "sc")
    monkeypatch.setattr(attempt, "_squeue", (float("-inf"), frozenset()))


def test_a_run_without_a_folder_gets_one_named_for_this_process(parent: Path) -> None:
    with claimed(parent) as folder:
        assert folder.parent == parent
        assert folder.name.startswith(f"local__{HOST}__{os.getpid()}__")
        assert current(parent) == folder
        assert why_held(parent) is not None
        (folder / "log").write_text("work")

    # Given back under a new name, with what was written in it.
    (free,) = parent.iterdir()
    assert free.name.startswith("free__")
    assert (free / "log").read_text() == "work"
    assert why_held(parent) is None


def test_a_folder_nothing_was_written_in_is_removed_with_its_run(parent: Path) -> None:
    with claimed(parent):
        pass

    assert not parent.exists()
    assert current(parent) == parent


def test_a_process_in_a_slurm_job_names_the_folder_for_the_job(
    parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    monkeypatch.setattr(attempt, "_cluster_name", "sc")

    with claimed(parent) as folder:
        assert folder.name.startswith("slurm__sc__7__2__")


def test_a_free_folder_is_taken_with_what_is_in_it(parent: Path) -> None:
    _folder(parent, "free__0")

    with claimed(parent) as folder:
        assert (folder / "log").read_text() == "earlier work"
        assert [path.name for path in parent.iterdir()] == [folder.name]


def test_the_folder_of_a_process_that_is_gone_is_taken(parent: Path) -> None:
    _folder(parent, f"local__{HOST}__{_dead_pid()}__0")

    with claimed(parent) as folder:
        assert (folder / "log").read_text() == "earlier work"


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("local__{host}__{pid}__0", "which is running"),
        ("local__elsewhere__1__0", "which only elsewhere can look for"),
        ("slurm__other__7__0__0", "which is not this cluster"),
        ("03be", "nobody furu knows"),
        ("free", "nobody furu knows"),
        ("local__host__notapid__0", "nobody furu knows"),
    ],
)
def test_a_held_unknown_or_misnamed_folder_is_refused_and_gets_no_sibling(
    parent: Path, monkeypatch: pytest.MonkeyPatch, name: str, why: str
) -> None:
    monkeypatch.setattr(attempt, "_cluster_name", "sc")
    name = name.format(host=HOST, pid=os.getpid())
    held = _folder(parent, name)

    with pytest.raises(Held, match=why) as refused, claimed(parent):
        pass

    # The error says which folder to free, and how.
    assert f"{held} is held by" in str(refused.value)
    assert f"rename it to {parent / 'free__0'}" in str(refused.value)
    assert [path.name for path in parent.iterdir()] == [name]


@pytest.mark.parametrize(
    ("queued", "states", "why"),
    [
        ("7\n", "COMPLETED\n", "queued or running"),
        ("", "", "accounting has no row for"),
        ("", "RUNNING\n", "accounting calls RUNNING"),
        ("", "PREEMPTED\nNODE_FAIL\n", "accounting calls NODE_FAIL"),
        ("", "BOOT_FAIL\n", "accounting calls BOOT_FAIL"),
        ("", "Kill task failed\n", "accounting calls Kill task failed"),
        ("", "PREEMPTED\nCOMPLETED\n", None),
        ("8\n77\n", "CANCELLED by 1000\n", None),
    ],
)
def test_a_slurm_jobs_folder_is_taken_only_once_squeue_and_every_sacct_row_agree(
    parent: Path,
    monkeypatch: pytest.MonkeyPatch,
    queued: str,
    states: str,
    why: str | None,
) -> None:
    _slurm(monkeypatch, queued=queued, states=states)
    _folder(parent, "slurm__sc__7__0__0")

    if why is None:
        with claimed(parent) as folder:
            assert (folder / "log").read_text() == "earlier work"
    else:
        with pytest.raises(Held, match=why), claimed(parent):
            pass
        assert [path.name for path in parent.iterdir()] == ["slurm__sc__7__0__0"]


def test_a_slurm_jobs_folder_is_held_while_slurm_does_not_answer(
    parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(*argv: str) -> str:
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(attempt, "_slurm", down)
    monkeypatch.setattr(attempt, "_cluster_name", "sc")
    _folder(parent, "slurm__sc__7__0__0")

    assert "could not be asked" in (why_held(parent) or "")


def test_a_rename_that_another_process_won_is_not_taken_for_a_claim(
    parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(parent, "free__0")
    rename = os.rename

    def lose(source: Path, target: Path) -> None:
        rename(source, parent / f"local__{HOST}__{os.getpid()}__winner")
        raise FileNotFoundError(source)

    monkeypatch.setattr(os, "rename", lose)

    assert attempt._try_claim(parent) == f"{parent} was taken by another process"
    assert [path.name for path in parent.iterdir()] == [
        f"local__{HOST}__{os.getpid()}__winner"
    ]


def test_a_rename_whose_answer_was_lost_is_a_claim(
    parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(parent, "free__0")
    rename = os.rename

    def done_but_unanswered(source: Path, target: Path) -> None:
        rename(source, target)
        raise OSError("the answer was lost")

    monkeypatch.setattr(os, "rename", done_but_unanswered)

    folder = attempt._try_claim(parent)

    assert isinstance(folder, Path)
    assert [path.name for path in parent.iterdir()] == [folder.name]


def test_an_empty_second_folder_is_removed_and_a_second_one_with_work_is_refused(
    parent: Path,
) -> None:
    _folder(parent, "free__0")
    (parent / "free__seed").mkdir()

    assert attempt._try_claim(parent) == f"{parent} has 2 folders where a run has one"
    with claimed(parent) as folder:
        assert (folder / "log").read_text() == "earlier work"

    _folder(parent, "free__1")
    with pytest.raises(Held, match="has 2 folders"), claimed(parent):
        pass
    assert len(list(parent.iterdir())) == 2


def test_a_dotted_folder_is_no_working_folder_and_goes_with_the_next_claim(
    parent: Path,
) -> None:
    _folder(parent, ".trash__0")
    assert current(parent) == parent
    assert why_held(parent) is None

    with claimed(parent) as folder:
        assert [path.name for path in parent.iterdir()] == [folder.name]


def test_discard_removes_a_folder_and_ignores_one_that_is_not_there(
    parent: Path,
) -> None:
    scratch = _folder(parent, "free__0") / "scratch"
    scratch.mkdir()

    attempt.discard(scratch, parent)
    attempt.discard(scratch, parent)

    assert [path.name for path in parent.iterdir()] == ["free__0"]
    assert not scratch.exists()


def test_short_lock_is_one_file_that_the_next_holder_waits_for(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"

    with short_lock(path):
        assert path.exists()
        with (
            pytest.raises(TimeoutError, match=f"{path} has been held"),
            short_lock(path, timeout_s=0.0),
        ):
            pass
        assert path.exists()

    assert not path.exists()
    with short_lock(path):
        pass
