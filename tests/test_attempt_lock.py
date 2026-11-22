"""The name of a run's working folder is its lock: who may take the folder,
how it is taken and given up, and what Slurm has to say before a job's is."""

import os
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from furu.storage import attempt_lock
from furu.storage.attempt_lock import AttemptHeld, Owner
from furu.worker.backends.slurm import command

HOST, PID = socket.gethostname(), os.getpid()
GONE = "slurm_load_jobs error: Invalid job id specified\n"


def _failed(argv: tuple[str, ...], stderr: str = "") -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(1, argv, stderr=stderr)


@pytest.fixture
def slurm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[tuple[str, ...]]]:
    """Answer Slurm's commands from a table: text is printed, an exception raised.

    A command with no answer is one the machine does not have. Returns the
    commands that were run.
    """

    def install(
        cluster: str | None = "sc", **answers: str | Exception
    ) -> list[tuple[str, ...]]:
        ran: list[tuple[str, ...]] = []

        def fake(*argv: str) -> str:
            ran.append(argv)
            answer = answers.get(argv[0], FileNotFoundError(argv[0]))
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr(command, "slurm", fake)
        monkeypatch.setattr(command, "cluster", lambda: cluster)
        return ran

    return install


# --- names -------------------------------------------------------------------


def test_this_process_is_named_by_host_and_pid_outside_slurm() -> None:
    assert attempt_lock._me() == f"local__{HOST}__{PID}"


@pytest.mark.parametrize(("restart", "named"), [(None, "0"), ("3", "3")])
def test_this_process_is_named_by_its_slurm_job_and_restart_count(
    monkeypatch: pytest.MonkeyPatch, restart: str | None, named: str
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "18609579")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    if restart is not None:
        monkeypatch.setenv("SLURM_RESTART_COUNT", restart)

    assert attempt_lock._me() == f"slurm__sc__18609579__{named}"


def test_a_free_folder_is_adoptable_and_a_local_one_is_held() -> None:
    assert attempt_lock.classify("free__seed") is Owner.ADOPTABLE
    assert attempt_lock.classify("free__9f3c2a7d51b0e6a4") is Owner.ADOPTABLE
    assert attempt_lock.classify("local__jagupard26__4242__9f3c") is Owner.HELD
    # A host name can hold dots.
    assert attempt_lock.classify("local__node.example.org__7__9f3c") is Owner.HELD


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
        "data",
        "3f2a9c",
        "",
    ],
)
def test_a_name_of_another_shape_is_never_adoptable(
    name: str, slurm: Callable[..., list[tuple[str, ...]]]
) -> None:
    ran = slurm(squeue=_failed(("squeue",), GONE), sacct="123|COMPLETED|None\n")

    assert attempt_lock.classify(name) is Owner.UNKNOWN
    assert ran == []  # nothing from the name reaches a command line


# --- what Slurm has to say ---------------------------------------------------

JOB = "slurm__sc__123__0__9f3c"


def test_a_job_gone_from_the_queue_that_ended_cleanly_is_adoptable(
    slurm: Callable[..., list[tuple[str, ...]]],
) -> None:
    ran = slurm(squeue=_failed(("squeue",), GONE), sacct="123|COMPLETED|None\n")

    assert attempt_lock.classify(JOB) is Owner.ADOPTABLE
    assert ran == [
        ("squeue", "--all", "--noheader", "-O", "JobID:32,State:32", "-j", "123"),
        (
            *("sacct", "-D", "-X", "--noheader", "--parsable2"),
            *("-o", "JobIDRaw,State,Reason", "-j", "123"),
        ),
    ]


@pytest.mark.parametrize(
    "accounted",
    [
        "123|FAILED|None\n",
        "123|CANCELLED by 24928|None\n",
        "123|TIMEOUT|None\n",
        "123|OUT_OF_MEMORY|None\n",
        "123|DEADLINE|None\n",
        "123|REVOKED|None\n",
        # Each time a requeued job ran.
        "123|PREEMPTED|None\n123|PREEMPTED|BeginTime\n123|COMPLETED|None\n",
    ],
)
def test_every_clean_end_is_adoptable(
    accounted: str, slurm: Callable[..., list[tuple[str, ...]]]
) -> None:
    # The queue can also answer with nothing for a job it has forgotten.
    slurm(squeue="", sacct=accounted)

    assert attempt_lock.classify(JOB) is Owner.ADOPTABLE


@pytest.mark.parametrize(
    "queued",
    [
        "123                             RUNNING                         \n",
        "123                             PENDING                         \n",
        "123                             COMPLETING                      \n",
        "123                             STAGE_OUT                       \n",
        "something this code has never seen\n",
    ],
)
def test_a_job_the_queue_still_lists_holds_its_folder(
    queued: str, slurm: Callable[..., list[tuple[str, ...]]]
) -> None:
    ran = slurm(squeue=queued, sacct="123|COMPLETED|None\n")

    assert attempt_lock.classify(JOB) is Owner.HELD
    assert [argv[0] for argv in ran] == ["squeue"]  # accounting is not asked


@pytest.mark.parametrize(
    "accounted",
    [
        "",  # accounting is behind the queue
        "123|NODE_FAIL|None\n",
        "123|BOOT_FAIL|None\n",
        "123|COMPLETED|Kill task failed\n",
        "123|RUNNING|None\n",
        "123|REQUEUED|None\n123|COMPLETED|None\n",
        "123|NODE_FAIL|None\n123|COMPLETED|None\n",
        "123|SOMETHING_NEW|None\n",
        "124|COMPLETED|None\n",  # another job's row only
        "123|COMPLETED\n",
        "not a row at all\n",
    ],
)
def test_an_end_that_is_not_known_to_be_clean_is_unknown(
    accounted: str, slurm: Callable[..., list[tuple[str, ...]]]
) -> None:
    slurm(squeue=_failed(("squeue",), GONE), sacct=accounted)

    assert attempt_lock.classify(JOB) is Owner.UNKNOWN


@pytest.mark.parametrize(
    "answers",
    [
        # The queue fails for another reason than a forgotten job.
        {
            "squeue": _failed(("squeue",), "slurm_load_jobs error: Socket timed out"),
            "sacct": "123|COMPLETED|None\n",
        },
        {
            "squeue": subprocess.TimeoutExpired("squeue", 60.0),
            "sacct": "123|COMPLETED|None\n",
        },
        {"squeue": "", "sacct": _failed(("sacct",), "sacct: error")},
        {"squeue": "", "sacct": subprocess.TimeoutExpired("sacct", 60.0)},
        {},  # a machine without Slurm
    ],
)
def test_slurm_that_does_not_answer_is_unknown(
    answers: dict[str, str | Exception], slurm: Callable[..., list[tuple[str, ...]]]
) -> None:
    slurm(**answers)

    assert attempt_lock.classify(JOB) is Owner.UNKNOWN


@pytest.mark.parametrize("cluster", ["other", None])
def test_a_job_of_another_cluster_is_unknown_and_slurm_is_not_asked(
    cluster: str | None, slurm: Callable[..., list[tuple[str, ...]]]
) -> None:
    ran = slurm(cluster, squeue="", sacct="123|COMPLETED|None\n")

    assert attempt_lock.classify(JOB) is Owner.UNKNOWN
    assert ran == []


def test_the_cluster_is_read_from_the_environment_or_asked_of_the_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    assert command.cluster() == "sc"

    monkeypatch.delenv("SLURM_CLUSTER_NAME")
    config = "Configuration data as of now\nClusterName             = other\nX = 1\n"
    monkeypatch.setattr(command, "slurm", lambda *argv: config)
    assert command.cluster() == "other"

    def missing(*argv: str) -> str:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(command, "slurm", missing)
    assert command.cluster() is None


# --- taking and giving up a folder -------------------------------------------


def test_the_first_worker_makes_the_folder_and_takes_it(tmp_path: Path) -> None:
    parent = tmp_path / "attempts" / "3f2a"

    attempt = attempt_lock.acquire(parent)

    assert attempt.parent == parent
    assert attempt.name.startswith(f"local__{HOST}__{PID}__")
    assert len(attempt.name.rpartition("__")[2]) == 16  # 64 bits of hex
    assert os.listdir(parent) == [attempt.name]


def test_a_released_folder_keeps_its_files_under_a_new_free_name(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "3f2a"
    first = attempt_lock.acquire(parent)
    (first / "checkpoint").write_text("step 7")

    attempt_lock.release(first)
    (freed,) = attempt_lock.entries(parent)
    second = attempt_lock.acquire(parent)
    attempt_lock.release(second)
    (freed_again,) = attempt_lock.entries(parent)

    assert freed.startswith("free__") and freed_again.startswith("free__")
    # No name is used twice.
    assert len({first.name, freed, second.name, freed_again, "free__seed"}) == 5
    assert (parent / freed_again / "checkpoint").read_text() == "step 7"


def test_a_folder_another_process_holds_is_refused_and_says_what_to_do(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "3f2a"
    held = parent / "local__jagupard26__4242__9f3c2a7d51b0e6a4"
    held.mkdir(parents=True)

    with pytest.raises(AttemptHeld) as refused:
        attempt_lock.acquire(parent)

    message = str(refused.value)
    assert f"{held} is the working folder of process 4242 on jagupard26" in message
    assert f"`mv {held} {parent / 'free__manual'}`" in message
    assert f"`rm -r {held}`" in message
    assert os.listdir(parent) == [held.name]  # nothing was made beside it

    # Done by hand, as the message says.
    held.rename(parent / "free__manual")
    assert attempt_lock.acquire(parent).name.startswith("local__")


def test_this_process_is_refused_its_own_folder_too(tmp_path: Path) -> None:
    parent = tmp_path / "3f2a"
    attempt_lock.acquire(parent)

    with pytest.raises(AttemptHeld, match=f"process {PID} on {HOST}"):
        attempt_lock.acquire(parent)


@pytest.mark.parametrize("stray", ["data", "local__host__4242", "3f2a.tmp"])
def test_a_folder_whose_name_says_nothing_is_refused_and_never_stepped_around(
    tmp_path: Path, stray: str
) -> None:
    parent = tmp_path / "3f2a"
    (parent / stray).mkdir(parents=True)

    with pytest.raises(AttemptHeld, match="its name does not say whose it is"):
        attempt_lock.acquire(parent)

    assert os.listdir(parent) == [stray]


def test_one_folder_that_is_not_free_refuses_the_run_even_beside_a_free_one(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "3f2a"
    (parent / "free__manual").mkdir(parents=True)
    (parent / "local__jagupard26__4242__9f3c").mkdir()

    with pytest.raises(AttemptHeld, match="process 4242 on jagupard26"):
        attempt_lock.acquire(parent)

    assert sorted(os.listdir(parent)) == [
        "free__manual",
        "local__jagupard26__4242__9f3c",
    ]


def test_folders_on_their_way_to_being_deleted_are_not_working_folders(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "3f2a"
    (parent / ".trash__9f3c").mkdir(parents=True)
    (parent / ".deleting__51b0").mkdir()

    assert attempt_lock.entries(parent) == []
    assert attempt_lock.current(parent) is None
    assert attempt_lock.blocked(parent) is None
    attempt = attempt_lock.acquire(parent)

    assert attempt_lock.current(parent) == attempt
    assert sorted(os.listdir(parent)) == sorted(
        [".trash__9f3c", ".deleting__51b0", attempt.name]
    )


def test_a_slurm_jobs_folder_is_taken_once_the_job_ended_cleanly(
    tmp_path: Path, slurm: Callable[..., list[tuple[str, ...]]]
) -> None:
    parent = tmp_path / "3f2a"
    (parent / JOB).mkdir(parents=True)
    (parent / JOB / "checkpoint").write_text("step 7")

    slurm(squeue="123    RUNNING\n")
    with pytest.raises(AttemptHeld, match="Slurm job 123, which is still queued"):
        attempt_lock.acquire(parent)

    slurm(squeue="", sacct="123|NODE_FAIL|None\n")
    with pytest.raises(AttemptHeld) as refused:
        attempt_lock.acquire(parent)
    assert "Slurm does not say from here that the job ended" in str(refused.value)
    assert f"`mv {parent / JOB} {parent / 'free__manual'}`" in str(refused.value)

    slurm(squeue="", sacct="123|PREEMPTED|None\n123|TIMEOUT|None\n")
    attempt = attempt_lock.acquire(parent)

    assert (attempt / "checkpoint").read_text() == "step 7"
    assert os.listdir(parent) == [attempt.name]


def test_of_two_workers_that_find_the_folder_free_one_takes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "3f2a"
    (parent / "free__seed").mkdir(parents=True)
    rename = os.rename
    theirs = parent / "local__jagupard26__4242__9f3c"

    def lose(source: Path, target: Path) -> None:
        # The other worker's rename lands between this one's listing and rename.
        rename(source, theirs)
        rename(source, target)

    monkeypatch.setattr(attempt_lock.os, "rename", lose)

    with pytest.raises(AttemptHeld, match="process 4242 on jagupard26"):
        attempt_lock.acquire(parent)

    assert os.listdir(parent) == [theirs.name]


def test_a_rename_that_happened_but_reported_an_error_is_found_in_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "3f2a"
    rename = os.rename
    calls = 0

    def answer_lost(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        rename(source, target)
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(attempt_lock.os, "rename", answer_lost)

    attempt = attempt_lock.acquire(parent)

    assert calls == 1
    assert os.listdir(parent) == [attempt.name]


def test_a_rename_that_keeps_failing_is_given_up_on_with_its_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "3f2a"

    def fail(source: Path, target: Path) -> None:
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(attempt_lock.os, "rename", fail)

    with pytest.raises(AttemptHeld, match="could not take a working folder") as gave_up:
        attempt_lock.acquire(parent)

    assert isinstance(gave_up.value.__cause__, OSError)
    assert os.listdir(parent) == ["free__seed"]  # still free for the next worker


def test_releasing_a_folder_that_was_taken_by_hand_does_nothing(tmp_path: Path) -> None:
    parent = tmp_path / "3f2a"
    attempt = attempt_lock.acquire(parent)
    attempt.rename(parent / "free__manual")

    attempt_lock.release(attempt)

    assert os.listdir(parent) == ["free__manual"]


def test_a_worker_frees_the_folder_its_dead_child_held_and_no_other(
    tmp_path: Path,
) -> None:
    host = socket.gethostname()
    childs, anothers = tmp_path / "3f2a", tmp_path / "77c1"
    (childs / f"local__{host}__4242__9f3c2a7d51b0e6a4").mkdir(parents=True)
    (childs / f"local__{host}__4242__9f3c2a7d51b0e6a4" / "checkpoint").write_text("7")
    # Process 42421 is not process 4242.
    (anothers / f"local__{host}__42421__0b1c2d3e4f5a6b7c").mkdir(parents=True)

    attempt_lock.release_child(childs, 4242)
    attempt_lock.release_child(anothers, 4242)

    (freed,) = attempt_lock.entries(childs)
    assert attempt_lock.classify(freed) is Owner.ADOPTABLE
    assert (childs / freed / "checkpoint").read_text() == "7"
    assert os.listdir(anothers) == [f"local__{host}__42421__0b1c2d3e4f5a6b7c"]


def test_a_slurm_workers_dead_child_held_the_folder_as_the_workers_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "18609579")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    parent = tmp_path / "3f2a"
    (parent / "slurm__sc__18609579__0__9f3c2a7d51b0e6a4").mkdir(parents=True)

    attempt_lock.release_child(parent, 4242)

    (freed,) = attempt_lock.entries(parent)
    assert freed.startswith("free__")
