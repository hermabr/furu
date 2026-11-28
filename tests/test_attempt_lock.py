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


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """Where a run's working folder goes: attempts/<run>, not made yet."""
    return tmp_path / "attempts" / "3f2a"


@pytest.fixture
def job_is(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[bool], list[tuple[str, str, int]]]:
    """Say whether Slurm calls a job's run gone. Returns the runs asked about."""

    def install(gone: bool) -> list[tuple[str, str, int]]:
        asked: list[tuple[str, str, int]] = []

        def fake(cluster: str, job: str, restart: int) -> bool:
            asked.append((cluster, job, restart))
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
    assert attempt_lock._me() == f"local__{HOST}__{PID}"


@pytest.mark.parametrize(("restart", "named"), [(None, "0"), ("", "0"), ("3", "3")])
def test_this_process_is_named_by_its_slurm_job_and_restart_count(
    monkeypatch: pytest.MonkeyPatch, restart: str | None, named: str
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "18609579")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    if restart is not None:
        monkeypatch.setenv("SLURM_RESTART_COUNT", restart)

    assert attempt_lock._me() == f"slurm__sc__18609579__{named}"


def test_a_slurm_job_that_cannot_tell_its_cluster_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nobody could ask Slurm about a job without knowing which Slurm to ask.
    monkeypatch.setenv("SLURM_JOB_ID", "18609579")
    monkeypatch.setattr(slurm, "cluster", lambda: None)

    with pytest.raises(RuntimeError, match="SLURM_JOB_ID=18609579.*SLURM_CLUSTER_NAME"):
        attempt_lock._me()


@pytest.mark.parametrize(
    "env",
    [
        {"SLURM_JOB_ID": "not-a-job"},
        {"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "sc", "SLURM_RESTART_COUNT": "x"},
    ],
)
def test_me_rejects_non_numeric_slurm_ids(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        attempt_lock._me()


@pytest.mark.parametrize("name", ["free__manual", "free__9f3c2a7d51b0e6a4"])
def test_a_free_folder_holds_nothing(run: Path, name: str) -> None:
    (run / name).mkdir(parents=True)

    assert attempt_lock.free(run) == run / name
    assert attempt_lock.current(run) == run / name


def test_a_run_without_a_working_folder_holds_nothing(run: Path) -> None:
    assert attempt_lock.free(run) is None
    assert attempt_lock.current(run) is None
    run.mkdir(parents=True)
    assert attempt_lock.free(run) is None
    assert attempt_lock.current(run) is None


# A host name can hold dots.
@pytest.mark.parametrize("name", [THEIRS, "local__node.example.org__7__9f3c"])
def test_a_process_holds_its_folder_whatever_became_of_it(run: Path, name: str) -> None:
    (run / name).mkdir(parents=True)
    host, pid = name.split("__")[1:3]

    with pytest.raises(HeldError) as refused:
        attempt_lock.free(run)

    why = str(refused.value)
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
    run: Path, name: str, job_is: Callable[[bool], list[tuple[str, str, int]]]
) -> None:
    asked = job_is(True)
    (run / name).mkdir(parents=True)

    with pytest.raises(HeldError, match="does not say whose working folder it is"):
        attempt_lock.free(run)
    with pytest.raises(HeldError, match="does not say whose working folder it is"):
        attempt_lock.acquire(run)
    assert asked == []  # nothing from the name reaches a command line
    assert os.listdir(run) == [name]


def test_a_slurm_jobs_folder_is_held_until_slurm_calls_the_job_gone(
    run: Path, job_is: Callable[[bool], list[tuple[str, str, int]]]
) -> None:
    (run / JOB).mkdir(parents=True)
    (run / JOB / "checkpoint").write_text("step 7")

    asked = job_is(False)
    with pytest.raises(HeldError) as refused:
        attempt_lock.free(run)
    why = str(refused.value)
    assert (
        f"{run / JOB} is the working folder of run 0 of Slurm job 123 on cluster sc"
        in why
    )
    assert f"mv {run / JOB} {run}/free__manual" in why
    with pytest.raises(HeldError, match="Slurm job 123 on cluster sc"):
        attempt_lock.acquire(run)
    assert asked == [("sc", "123", 0)] * 2
    assert os.listdir(run) == [JOB]

    job_is(True)
    assert attempt_lock.free(run) == run / JOB
    attempt, _ = attempt_lock.acquire(run)

    assert (attempt / "checkpoint").read_text() == "step 7"
    assert os.listdir(run) == [attempt.name]


def test_a_requeued_job_takes_the_folder_its_last_run_held(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run / JOB).mkdir(parents=True)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    # The queue runs the job again; accounting has a row for each run.
    answers = {"squeue": "123 RUNNING 1\n", "sacct": "123|REQUEUED|None\n"}
    monkeypatch.setattr(slurm, "run", lambda *argv: answers[argv[0]])

    attempt, _ = attempt_lock.acquire(run)

    assert attempt.name.startswith("slurm__sc__123__1__")
    # The run that holds it now is the one that runs.
    with pytest.raises(HeldError, match="run 1 of Slurm job 123"):
        attempt_lock.free(run)


def test_two_folders_hold_the_run_however_they_are_named(run: Path) -> None:
    (run / "free__manual").mkdir(parents=True)
    (run / "free__other").mkdir()

    with pytest.raises(HeldError) as refused:
        attempt_lock.free(run)

    assert f"{run} holds 2 folders, free__manual, free__other" in str(refused.value)
    with pytest.raises(HeldError, match="holds 2 folders"):
        attempt_lock.acquire(run)
    assert sorted(os.listdir(run)) == ["free__manual", "free__other"]


# --- taking a folder ---------------------------------------------------------


def test_the_first_worker_makes_the_folder_and_takes_it(run: Path) -> None:
    attempt, claimed = attempt_lock.acquire(run)

    assert attempt.parent == run
    assert attempt.name.startswith(f"local__{HOST}__{PID}__")
    assert len(attempt.name.rpartition("__")[2]) == 16  # 64 bits of hex
    assert os.path.samestat(claimed, attempt.stat())
    assert os.listdir(run) == [attempt.name]
    assert os.listdir(run.parent) == [run.name]  # nothing is left beside it


def test_a_folder_is_named_with_the_token_it_is_taken_with(run: Path) -> None:
    attempt, _ = attempt_lock.acquire(run, token="0b1c2d3e4f5a6b7c")
    assert attempt.name == f"local__{HOST}__{PID}__0b1c2d3e4f5a6b7c"

    (attempt / "checkpoint").write_text("step 1")
    attempt_lock.release(attempt)
    again, _ = attempt_lock.acquire(run, token="a1b2c3d4e5f60718")
    assert again.name == f"local__{HOST}__{PID}__a1b2c3d4e5f60718"


def test_a_folder_another_process_holds_is_refused_and_says_what_to_do(
    run: Path,
) -> None:
    held = run / THEIRS
    held.mkdir(parents=True)

    with pytest.raises(HeldError) as refused:
        attempt_lock.acquire(run)

    message = str(refused.value)
    assert f"{held} is the working folder of process 4242 on jagupard26" in message
    assert f"mv {held} {run / 'free__manual'}" in message
    assert os.listdir(run) == [held.name]
    assert os.listdir(run.parent) == [run.name]  # nothing was made beside it

    # Done by hand, as the message says.
    held.rename(run / "free__manual")
    assert attempt_lock.acquire(run)[0].name.startswith("local__")


def test_this_process_is_refused_its_own_folder_too(run: Path) -> None:
    attempt_lock.acquire(run)

    with pytest.raises(HeldError, match=f"process {PID} on {HOST}"):
        attempt_lock.acquire(run)


def test_of_the_workers_that_start_a_run_at_once_one_gets_its_folder(run: Path) -> None:
    workers = 8
    together = threading.Barrier(workers)
    got: list[Path] = []
    refused: list[HeldError] = []

    def start() -> None:
        together.wait()
        try:
            got.append(attempt_lock.acquire(run)[0])
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
        attempt_lock.acquire(run)

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
        attempt_lock.acquire(run)

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

    attempt, _ = attempt_lock.acquire(run)

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
        attempt_lock.acquire(run)

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

    attempt, _ = attempt_lock.acquire(run)

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

    with pytest.raises(OSError, match="Disk quota exceeded") as refused:
        attempt_lock.acquire(run)

    assert refused.value.errno == errno.EDQUOT
    assert list_names(run) == (["free__manual"] if there else [])
    assert os.listdir(run.parent) == ([run.name] if there else [])


def test_a_failing_stat_in_acquire_releases_fresh_or_adopted_claim(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_stat = Path.stat

    def boom(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self.parent == run:
            raise OSError(errno.ESTALE, "Stale file handle")
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", boom)

    # Fresh claim is removed rather than stranded under local__.
    with pytest.raises(OSError, match="Stale file handle"):
        attempt_lock.acquire(run)
    assert not run.exists()

    # Adopted claim with checkpoints is released back to free__ with contents intact.
    (run / "free__manual" / "scratch").mkdir(parents=True)
    (run / "free__manual" / "scratch" / "ckpt").write_text("step 5")
    with pytest.raises(OSError, match="Stale file handle"):
        attempt_lock.acquire(run)
    (freed,) = os.listdir(run)
    assert freed.startswith("free__") and freed != "free__manual"
    assert (run / freed / "scratch" / "ckpt").read_text() == "step 5"


# --- giving a folder up ------------------------------------------------------


def test_releasing_an_empty_folder_removes_it_with_its_parent(run: Path) -> None:
    attempt, _ = attempt_lock.acquire(run)

    attempt_lock.release(attempt)

    assert os.listdir(run.parent) == []


def test_a_released_folder_keeps_its_files_under_a_new_free_name(run: Path) -> None:
    first, _ = attempt_lock.acquire(run)
    (first / "checkpoint").write_text("step 7")

    attempt_lock.release(first)
    (freed,) = os.listdir(run)
    second, _ = attempt_lock.acquire(run)
    attempt_lock.release(second)
    (freed_again,) = os.listdir(run)

    assert freed.startswith("free__") and freed_again.startswith("free__")
    assert len({first.name, freed, second.name, freed_again}) == 4  # none used twice
    assert (run / freed_again / "checkpoint").read_text() == "step 7"


def test_releasing_a_folder_that_was_taken_by_hand_does_nothing(run: Path) -> None:
    attempt, _ = attempt_lock.acquire(run)
    attempt.rename(run / "free__manual")

    attempt_lock.release(attempt)

    assert os.listdir(run) == ["free__manual"]


def test_a_release_that_failed_says_that_the_folder_stays_held(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt, _ = attempt_lock.acquire(run)
    (attempt / "checkpoint").write_text("step 7")
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
    child, _ = attempt_lock.acquire(childs, token="9f3c2a7d51b0e6a4")
    (child / "checkpoint").write_text("7")
    sibling, _ = attempt_lock.acquire(siblings, token="0b1c2d3e4f5a6b7c")
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
    attempt, claimed = attempt_lock.acquire(run)
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
    attempt, _ = attempt_lock.acquire(run)
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


def test_a_failed_runs_mkdir_in_publish_leaves_scratch_in_place(
    run: Path, tmp_path: Path
) -> None:
    attempt, claimed = attempt_lock.acquire(run)
    (attempt / "scratch").mkdir()
    (attempt / "scratch" / "ckpt").write_text("step 9")
    # Make dest.parent uncreatable by putting a file where its parent would be.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file")

    with pytest.raises(OSError):
        attempt_lock.publish(attempt, blocker / "runs" / run.name, claimed)

    assert (attempt / "scratch" / "ckpt").read_text() == "step 9"
    assert os.listdir(run.parent) == [run.name]
