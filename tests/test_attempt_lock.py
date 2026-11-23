"""The name of a run's working folder is its lock: who may take the folder, and
how it is taken, given up, published and deleted."""

import errno
import os
import socket
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from furu import slurm
from furu.result import bundle
from furu.storage import attempt_lock
from furu.storage._layout import list_names
from furu.storage.attempt_lock import HeldError
from furu.storage.directory import SpecDirectory

HOST, PID = socket.gethostname(), os.getpid()
THEIRS = "local__jagupard26__4242__9f3c2a7d51b0e6a4"
JOB = "slurm__sc__123__0__9f3c2a7d51b0e6a4"


def unpublished() -> bool:
    return False


def _take(run: Path, token: str | None = None) -> tuple[Path, os.stat_result]:
    claim = attempt_lock.acquire(run, unpublished, token=token)
    assert claim is not None
    return claim


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """Where a run's working folder goes: attempts/<run>, not made yet."""
    return tmp_path / "attempts" / "3f2a"


@pytest.fixture
def job_is(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], list[tuple[str, str]]]:
    """Say whether Slurm calls a job gone. Returns the jobs that were asked about."""

    def install(gone: bool) -> list[tuple[str, str]]:
        asked: list[tuple[str, str]] = []

        def fake(cluster: str, job: str) -> bool:
            asked.append((cluster, job))
            return gone

        monkeypatch.setattr(slurm, "gone", fake)
        return asked

    return install


def _renames(
    monkeypatch: pytest.MonkeyPatch, rename: Callable[[Path, Path], None]
) -> None:
    monkeypatch.setattr(attempt_lock.os, "rename", rename)


# --- names -------------------------------------------------------------------


def test_this_process_is_named_by_host_and_pid_outside_slurm() -> None:
    assert attempt_lock.me() == f"local__{HOST}__{PID}"


@pytest.mark.parametrize(("restart", "named"), [(None, "0"), ("3", "3")])
def test_this_process_is_named_by_its_slurm_job_and_restart_count(
    monkeypatch: pytest.MonkeyPatch, restart: str | None, named: str
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "18609579")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    if restart is not None:
        monkeypatch.setenv("SLURM_RESTART_COUNT", restart)

    assert attempt_lock.me() == f"slurm__sc__18609579__{named}"


def test_a_slurm_job_that_cannot_tell_its_cluster_is_named_as_a_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nobody could ask Slurm about a job without knowing which Slurm to ask.
    monkeypatch.setenv("SLURM_JOB_ID", "18609579")
    monkeypatch.setattr(slurm, "cluster", lambda: None)

    assert attempt_lock.me() == f"local__{HOST}__{PID}"


@pytest.mark.parametrize("name", ["free__manual", "free__9f3c2a7d51b0e6a4"])
def test_a_free_folder_holds_nothing(run: Path, name: str) -> None:
    (run / name).mkdir(parents=True)

    assert attempt_lock.held(run) is None
    assert attempt_lock.current(run) == run / name


def test_a_run_without_a_working_folder_holds_nothing(run: Path) -> None:
    assert attempt_lock.held(run) is None
    assert attempt_lock.current(run) is None
    run.mkdir(parents=True)
    assert attempt_lock.held(run) is None
    assert attempt_lock.current(run) is None


# A host name can hold dots.
@pytest.mark.parametrize("name", [THEIRS, "local__node.example.org__7__9f3c"])
def test_a_process_holds_its_folder_whatever_became_of_it(run: Path, name: str) -> None:
    (run / name).mkdir(parents=True)
    host, pid = name.split("__")[1:3]

    why = attempt_lock.held(run)

    assert why is not None
    assert f"{run / name} is the working folder of process {pid} on {host}" in why
    assert why.endswith(f"free it by hand: mv {run / name} {run}/free__manual")


@pytest.mark.parametrize(
    "name",
    [
        "free",
        "free__",
        "free__a__b",
        "local__host__4242",
        "local__host__notapid__9f3c",
        "local__host__4242__9f3c__extra",
        "local____4242__9f3c",
        "slurm__sc__123__0",
        "slurm__sc__123__0__9f3c__extra",
        "slurm__sc__12,13__0__9f3c",
        "slurm__sc__--all__0__9f3c",
        "slurm__sc__123__x__9f3c",
        "slurm____123__0__9f3c",
        "data",
        ".trash__9f3c",
    ],
)
def test_a_name_of_another_shape_holds_the_run_and_slurm_is_not_asked(
    run: Path, name: str, job_is: Callable[[bool], list[tuple[str, str]]]
) -> None:
    asked = job_is(True)
    (run / name).mkdir(parents=True)

    why = attempt_lock.held(run)

    assert why is not None and "does not say whose working folder it is" in why
    with pytest.raises(HeldError, match="does not say whose working folder it is"):
        attempt_lock.acquire(run, unpublished)
    assert asked == []  # nothing from the name reaches a command line
    assert os.listdir(run) == [name]


def test_a_slurm_jobs_folder_is_held_until_slurm_calls_the_job_gone(
    run: Path, job_is: Callable[[bool], list[tuple[str, str]]]
) -> None:
    (run / JOB).mkdir(parents=True)
    (run / JOB / "checkpoint").write_text("step 7")

    asked = job_is(False)
    why = attempt_lock.held(run)
    assert why is not None
    assert f"{run / JOB} is the working folder of Slurm job 123 on cluster sc" in why
    assert f"mv {run / JOB} {run}/free__manual" in why
    with pytest.raises(HeldError, match="Slurm job 123 on cluster sc"):
        attempt_lock.acquire(run, unpublished)
    assert asked == [("sc", "123")] * 2
    assert os.listdir(run) == [JOB]

    job_is(True)
    assert attempt_lock.held(run) is None
    attempt, _ = _take(run)

    assert (attempt / "checkpoint").read_text() == "step 7"
    assert os.listdir(run) == [attempt.name]


def test_two_folders_hold_the_run_however_they_are_named(run: Path) -> None:
    (run / "free__manual").mkdir(parents=True)
    (run / "free__other").mkdir()

    why = attempt_lock.held(run)

    assert why is not None
    assert f"{run} holds 2 folders, free__manual, free__other" in why
    with pytest.raises(HeldError, match="holds 2 folders"):
        attempt_lock.acquire(run, unpublished)
    assert sorted(os.listdir(run)) == ["free__manual", "free__other"]


# --- taking a folder ---------------------------------------------------------


def test_the_first_worker_makes_the_folder_and_takes_it(run: Path) -> None:
    attempt, claimed = _take(run)

    assert attempt.parent == run
    assert attempt.name.startswith(f"local__{HOST}__{PID}__")
    assert len(attempt.name.rpartition("__")[2]) == 16  # 64 bits of hex
    assert os.path.samestat(claimed, attempt.stat())
    assert os.listdir(run) == [attempt.name]
    assert os.listdir(run.parent) == [run.name]  # nothing is left beside it


def test_a_folder_is_named_with_the_token_it_is_taken_with(run: Path) -> None:
    attempt, _ = _take(run, token="0b1c2d3e4f5a6b7c")
    assert attempt.name == f"local__{HOST}__{PID}__0b1c2d3e4f5a6b7c"

    attempt_lock.release(attempt)
    again, _ = _take(run, token="a1b2c3d4e5f60718")
    assert again.name == f"local__{HOST}__{PID}__a1b2c3d4e5f60718"


def test_a_folder_another_process_holds_is_refused_and_says_what_to_do(
    run: Path,
) -> None:
    held = run / THEIRS
    held.mkdir(parents=True)

    with pytest.raises(HeldError) as refused:
        attempt_lock.acquire(run, unpublished)

    message = str(refused.value)
    assert f"{held} is the working folder of process 4242 on jagupard26" in message
    assert f"mv {held} {run / 'free__manual'}" in message
    assert os.listdir(run) == [held.name]
    assert os.listdir(run.parent) == [run.name]  # nothing was made beside it

    # Done by hand, as the message says.
    held.rename(run / "free__manual")
    assert _take(run)[0].name.startswith("local__")


def test_this_process_is_refused_its_own_folder_too(run: Path) -> None:
    _take(run)

    with pytest.raises(HeldError, match=f"process {PID} on {HOST}"):
        attempt_lock.acquire(run, unpublished)


def test_of_the_workers_that_start_a_run_at_once_one_gets_its_folder(run: Path) -> None:
    workers = 8
    together = threading.Barrier(workers)
    got: list[Path] = []
    refused: list[HeldError] = []

    def start() -> None:
        together.wait()
        try:
            got.append(_take(run)[0])
        except HeldError as exc:
            refused.append(exc)

    threads = [threading.Thread(target=start) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    (winner,) = got
    assert len(refused) == workers - 1
    assert os.listdir(run) == [winner.name]
    assert os.listdir(run.parent) == [run.name]


def test_a_folder_made_too_late_does_not_land_beside_the_one_that_is_there(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rename = os.rename

    def late(source: Path, target: Path) -> None:
        # Another worker's folder lands between this one's listing and rename.
        (run / THEIRS).mkdir(parents=True)
        (run / THEIRS / "checkpoint").write_text("step 7")
        rename(source, target)

    _renames(monkeypatch, late)

    with pytest.raises(HeldError, match="did not get") as refused:
        attempt_lock.acquire(run, unpublished)

    assert f"Its working folder is now {run / THEIRS}" in str(refused.value)
    assert os.listdir(run) == [THEIRS]
    assert (run / THEIRS / "checkpoint").read_text() == "step 7"
    assert os.listdir(run.parent) == [run.name]


def test_of_two_workers_that_find_the_folder_free_one_takes_it(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run / "free__manual").mkdir(parents=True)
    rename = os.rename

    def lose(source: Path, target: Path) -> None:
        # The other worker's rename lands between this one's listing and rename.
        rename(source, run / THEIRS)
        rename(source, target)

    _renames(monkeypatch, lose)

    with pytest.raises(HeldError, match="did not get"):
        attempt_lock.acquire(run, unpublished)

    assert os.listdir(run) == [THEIRS]
    assert os.listdir(run.parent) == [run.name]


def test_a_listing_taken_during_a_rename_is_taken_again(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run / "free__manual").mkdir(parents=True)
    listings = 0

    def both_names_once(folder: Path) -> list[str]:
        nonlocal listings
        listings += 1
        names = list_names(folder)
        return [*names, THEIRS] if listings == 1 else names

    monkeypatch.setattr(attempt_lock, "list_names", both_names_once)

    attempt, _ = _take(run)

    assert os.listdir(run) == [attempt.name]


def test_a_run_that_keeps_listing_two_folders_is_given_up_on_after_three_listings(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run / "free__manual").mkdir(parents=True)
    listings: list[Path] = []

    def both_names(folder: Path) -> list[str]:
        listings.append(folder)
        return [*list_names(folder), THEIRS]

    monkeypatch.setattr(attempt_lock, "list_names", both_names)

    with pytest.raises(HeldError, match="holds 2 folders"):
        attempt_lock.acquire(run, unpublished)

    assert listings == [run] * 3
    assert os.listdir(run) == ["free__manual"]


@pytest.mark.parametrize("lost", [errno.ENOENT, errno.ESTALE, errno.EIO])
@pytest.mark.parametrize("there", [False, True], ids=["new", "free"])
def test_a_rename_that_happened_but_reported_an_error_is_found_in_the_listing(
    run: Path, monkeypatch: pytest.MonkeyPatch, lost: int, there: bool
) -> None:
    if there:
        (run / "free__manual").mkdir(parents=True)
        (run / "free__manual" / "checkpoint").write_text("step 7")
    rename = os.rename
    calls = 0

    def answer_lost(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        rename(source, target)
        raise OSError(lost, os.strerror(lost))

    _renames(monkeypatch, answer_lost)

    attempt, _ = _take(run)

    assert calls == 1
    assert os.listdir(run) == [attempt.name]
    assert os.listdir(attempt) == (["checkpoint"] if there else [])
    assert os.listdir(run.parent) == [run.name]


@pytest.mark.parametrize("there", [False, True], ids=["new", "free"])
def test_a_rename_that_failed_leaves_the_run_as_it_was(
    run: Path, monkeypatch: pytest.MonkeyPatch, there: bool
) -> None:
    if there:
        (run / "free__manual").mkdir(parents=True)

    def fail(source: Path, target: Path) -> None:
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    _renames(monkeypatch, fail)

    with pytest.raises(HeldError, match="did not get") as refused:
        attempt_lock.acquire(run, unpublished)

    assert ("is now" if there else "is gone") in str(refused.value)
    assert list_names(run) == (["free__manual"] if there else [])
    assert os.listdir(run.parent) == ([run.name] if there else [])


# --- a run that was published meanwhile --------------------------------------


def test_a_run_published_before_its_folder_was_made_is_not_started_again(
    run: Path,
) -> None:
    seen: list[list[str]] = []

    def published() -> bool:
        seen.append(os.listdir(run))  # asked once the folder is this process's
        return True

    assert attempt_lock.acquire(run, published, token="0b1c2d3e4f5a6b7c") is None

    assert seen == [[f"local__{HOST}__{PID}__0b1c2d3e4f5a6b7c"]]
    assert os.listdir(run.parent) == []  # the folder made here is gone again


def test_a_run_published_before_its_folder_was_taken_keeps_what_is_in_the_folder(
    run: Path,
) -> None:
    (run / "free__manual").mkdir(parents=True)
    (run / "free__manual" / "checkpoint").write_text("step 7")

    assert attempt_lock.acquire(run, lambda: True) is None

    (freed,) = os.listdir(run)
    assert freed.startswith("free__") and freed != "free__manual"
    assert (run / freed / "checkpoint").read_text() == "step 7"


# --- giving a folder up ------------------------------------------------------


def test_a_released_folder_keeps_its_files_under_a_new_free_name(run: Path) -> None:
    first, _ = _take(run)
    (first / "checkpoint").write_text("step 7")

    attempt_lock.release(first)
    (freed,) = os.listdir(run)
    second, _ = _take(run)
    attempt_lock.release(second)
    (freed_again,) = os.listdir(run)

    assert freed.startswith("free__") and freed_again.startswith("free__")
    assert len({first.name, freed, second.name, freed_again}) == 4  # none used twice
    assert (run / freed_again / "checkpoint").read_text() == "step 7"


def test_releasing_a_folder_that_was_taken_by_hand_does_nothing(run: Path) -> None:
    attempt, _ = _take(run)
    attempt.rename(run / "free__manual")

    attempt_lock.release(attempt)

    assert os.listdir(run) == ["free__manual"]


def test_a_release_that_failed_says_that_the_folder_stays_held(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt, _ = _take(run)
    warned: list[str] = []
    monkeypatch.setattr(
        attempt_lock.logger, "warning", lambda text, *args: warned.append(text % args)
    )

    def fail(source: Path, target: Path) -> None:
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    _renames(monkeypatch, fail)

    attempt_lock.release(attempt)

    (warning,) = warned
    assert f"could not release {attempt}, which stays held" in warning
    assert os.listdir(run) == [attempt.name]


def test_a_worker_frees_the_folder_its_dead_child_held_and_no_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two children of one Slurm job hold their folders under the same job:
    # only the token says which child a folder was.
    monkeypatch.setenv("SLURM_JOB_ID", "18609579")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    childs, siblings = tmp_path / "attempts" / "3f2a", tmp_path / "attempts" / "77c1"
    child, _ = _take(childs, token="9f3c2a7d51b0e6a4")
    (child / "checkpoint").write_text("7")
    sibling, _ = _take(siblings, token="0b1c2d3e4f5a6b7c")
    assert child.name == "slurm__sc__18609579__0__9f3c2a7d51b0e6a4"
    assert sibling.name == "slurm__sc__18609579__0__0b1c2d3e4f5a6b7c"

    attempt_lock.release_child(childs, "9f3c2a7d51b0e6a4")
    attempt_lock.release_child(siblings, "9f3c2a7d51b0e6a4")

    (freed,) = os.listdir(childs)
    assert freed.startswith("free__")
    assert (childs / freed / "checkpoint").read_text() == "7"
    assert os.listdir(siblings) == [sibling.name]


def test_a_child_that_died_before_it_took_a_folder_left_nothing_to_free(
    run: Path,
) -> None:
    attempt_lock.release_child(run, "9f3c2a7d51b0e6a4")
    assert not run.exists()

    # A folder that is free is nobody's, whatever its name ends with.
    (run / "free__9f3c2a7d51b0e6a4").mkdir(parents=True)
    attempt_lock.release_child(run, "9f3c2a7d51b0e6a4")
    assert os.listdir(run) == ["free__9f3c2a7d51b0e6a4"]


# --- publishing --------------------------------------------------------------


@pytest.fixture
def finished(run: Path) -> tuple[Path, Path, os.stat_result]:
    """A taken working folder with a result and scratch in it, and where its
    run is published."""
    attempt, claimed = _take(run)
    (attempt / "result").write_text("42")
    (attempt / "scratch").mkdir()
    (attempt / "scratch" / "tmp").write_text("left over")
    return attempt, run.parent.parent / "fixed" / "runs" / run.name, claimed


def test_a_finished_folder_becomes_the_run_without_its_scratch(
    run: Path, finished: tuple[Path, Path, os.stat_result]
) -> None:
    attempt, dest, claimed = finished

    attempt_lock.publish(attempt, dest, claimed)

    assert os.listdir(dest) == ["result"]
    assert os.listdir(run.parent) == []  # no working folder, and nothing on its way out


def test_a_publish_that_failed_leaves_the_folder_as_it_was(
    run: Path,
    finished: tuple[Path, Path, os.stat_result],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, dest, claimed = finished
    rename = os.rename

    def fail_to_publish(source: Path, target: Path) -> None:
        if target == dest:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        rename(source, target)

    _renames(monkeypatch, fail_to_publish)

    with pytest.raises(OSError, match="Disk quota exceeded"):
        attempt_lock.publish(attempt, dest, claimed)

    assert not dest.exists()
    assert sorted(os.listdir(attempt)) == ["result", "scratch"]
    assert (attempt / "scratch" / "tmp").read_text() == "left over"
    assert os.listdir(run.parent) == [run.name]
    # Still this process's, to release or to publish again.
    monkeypatch.undo()
    attempt_lock.publish(attempt, dest, claimed)
    assert os.listdir(dest) == ["result"]


@pytest.mark.parametrize("lost", [errno.ENOENT, errno.ESTALE, errno.EIO])
def test_a_publish_that_happened_but_reported_an_error_is_found_in_the_listing(
    run: Path,
    finished: tuple[Path, Path, os.stat_result],
    monkeypatch: pytest.MonkeyPatch,
    lost: int,
) -> None:
    attempt, dest, claimed = finished
    rename = os.rename

    def answer_lost(source: Path, target: Path) -> None:
        rename(source, target)
        if target == dest:
            raise OSError(lost, os.strerror(lost))

    _renames(monkeypatch, answer_lost)

    attempt_lock.publish(attempt, dest, claimed)

    assert os.listdir(dest) == ["result"]
    assert os.listdir(run.parent) == []


def test_a_run_that_is_already_published_is_not_replaced(
    run: Path, finished: tuple[Path, Path, os.stat_result]
) -> None:
    attempt, dest, claimed = finished
    dest.mkdir(parents=True)
    (dest / "result").write_text("the first one")

    attempt_lock.publish(attempt, dest, claimed)

    assert (dest / "result").read_text() == "the first one"
    assert os.listdir(run.parent) == []  # the folder that lost is deleted


@pytest.mark.parametrize("made_again", [False, True])
def test_a_folder_that_was_taken_from_its_process_is_not_published(
    run: Path, finished: tuple[Path, Path, os.stat_result], made_again: bool
) -> None:
    attempt, dest, claimed = finished
    attempt.rename(run / "free__manual")
    if made_again:
        # As a create() does that writes to a path with its parents.
        (attempt / "data").mkdir(parents=True)
        (attempt / "data" / "late").write_text("written after the folder was taken")

    with pytest.raises(HeldError, match="is no longer the folder this process took"):
        attempt_lock.publish(attempt, dest, claimed)

    assert not dest.exists()
    # The folder that was taken is untouched, and alone again.
    assert os.listdir(run) == ["free__manual"]
    assert sorted(os.listdir(run / "free__manual")) == ["result", "scratch"]
    assert os.listdir(run.parent) == [run.name]


def test_a_create_cannot_make_its_folder_again_through_its_directory(
    run: Path,
) -> None:
    attempt, _ = _take(run)
    directory = SpecDirectory(attempt)
    attempt.rename(run / "free__manual")

    with pytest.raises(FileNotFoundError):
        _ = directory.data
    with pytest.raises(FileNotFoundError):
        _ = directory.scratch
    with pytest.raises(FileNotFoundError):
        bundle._save_result_bundle(
            [42],
            attempt / "result",
            spec_classes=[object],
            data_dir=attempt / "data",
            label="a result",
        )
    with pytest.raises(FileNotFoundError):
        bundle._make_in(attempt / "result", Path("artifacts", "weights", "layer"))

    assert os.listdir(run) == ["free__manual"]


def test_a_directory_in_a_bundle_is_made_with_the_ones_above_it(tmp_path: Path) -> None:
    bundle._make_in(tmp_path, Path("rows", "0", "weights"))
    bundle._make_in(tmp_path, Path("rows", "1", "weights"))

    assert sorted(os.listdir(tmp_path / "rows")) == ["0", "1"]
    with pytest.raises(FileExistsError):
        bundle._make_in(tmp_path, Path("rows", "1", "weights"))


# --- deleting ----------------------------------------------------------------


def test_a_free_folder_is_deleted_with_what_held_it(run: Path) -> None:
    (run / "free__manual" / "data").mkdir(parents=True)

    attempt_lock.discard(run)

    assert os.listdir(run.parent) == []


def test_deleting_the_folder_of_a_run_that_has_none_does_nothing(run: Path) -> None:
    run.parent.mkdir()

    attempt_lock.discard(run)

    assert os.listdir(run.parent) == []


def test_a_folder_another_process_holds_is_not_deleted(run: Path) -> None:
    (run / THEIRS / "data").mkdir(parents=True)

    with pytest.raises(HeldError, match="process 4242 on jagupard26"):
        attempt_lock.discard(run)

    assert os.listdir(run / THEIRS) == ["data"]
    assert os.listdir(run.parent) == [run.name]


def test_a_folder_a_worker_took_first_is_not_deleted(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run / "free__manual" / "data").mkdir(parents=True)
    rename = os.rename

    def lose(source: Path, target: Path) -> None:
        # The worker's rename lands between this one's listing and rename.
        rename(source, run / THEIRS)
        rename(source, target)

    _renames(monkeypatch, lose)

    with pytest.raises(HeldError, match=f"was not deleted.*now holds {THEIRS}"):
        attempt_lock.discard(run)

    assert os.listdir(run / THEIRS) == ["data"]
    assert os.listdir(run.parent) == [run.name]
