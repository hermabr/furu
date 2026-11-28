"""How Slurm's commands are run, and what Slurm has to say before a job counts
as gone."""

import os
import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from furu import slurm

FORGOTTEN = "slurm_load_jobs error: Invalid job id specified\n"
SQUEUE = (
    *("squeue", "--all", "--array", "--noheader"),
    *("-O", "JobID:20,State:20,RestartCnt:10", "-j", "123"),
)
SACCT = (
    *("sacct", "-D", "-X", "--noheader", "--parsable2"),
    *("-o", "JobIDRaw,State,Reason", "-j", "123"),
)

Answers = Callable[..., list[tuple[str, ...]]]


def _failed(name: str, stderr: str = "", stdout: str = "") -> Exception:
    return subprocess.CalledProcessError(1, name, output=stdout, stderr=stderr)


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> Answers:
    """Answer Slurm's commands from a table: text is printed, an exception raised.

    A command with no answer is one the machine does not have. Returns the
    commands that were run.
    """

    def install(
        cluster: str | None = "sc", **table: str | Exception
    ) -> list[tuple[str, ...]]:
        ran: list[tuple[str, ...]] = []

        def fake(*argv: str) -> str:
            ran.append(argv)
            answer = table.get(argv[0], FileNotFoundError(argv[0]))
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr(slurm, "run", fake)
        monkeypatch.setattr(slurm, "cluster", lambda: cluster)
        return ran

    return install


# --- running a command -------------------------------------------------------


def test_a_command_runs_without_what_changes_its_output_or_its_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SQUEUE_FORMAT", "%i")
    monkeypatch.setenv("SQUEUE_STATES", "RUNNING")
    monkeypatch.setenv("SACCT_FORMAT", "JobID")
    monkeypatch.setenv("SLURM_CLUSTERS", "elsewhere")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
    seen: dict[str, Any] = {}

    def fake(argv: tuple[str, ...], **how: Any) -> subprocess.CompletedProcess[str]:
        seen.update(how, argv=argv)
        return subprocess.CompletedProcess(argv, 0, stdout="123 RUNNING\n")

    monkeypatch.setattr(slurm.subprocess, "run", fake)

    assert slurm.run("squeue", "-j", "123") == "123 RUNNING\n"

    environment = seen.pop("env")
    assert environment["SLURM_CONF"] == "/etc/slurm/slurm.conf"
    assert not {"SQUEUE_FORMAT", "SQUEUE_STATES", "SACCT_FORMAT", "SLURM_CLUSTERS"} & (
        environment.keys()
    )
    assert seen == {
        "argv": ("squeue", "-j", "123"),
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 60.0,
    }


def test_a_job_knows_its_cluster_from_its_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    monkeypatch.setattr(slurm, "run", lambda *argv: pytest.fail("asked"))

    assert slurm.cluster() == "sc"


def test_the_controller_is_asked_for_the_cluster_once_it_has_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = "Configuration data as of now\nClusterName             = other\nX = 1\n"
    answers: list[str | Exception] = [
        FileNotFoundError("scontrol"),
        subprocess.TimeoutExpired("scontrol", 60.0),
        "X = 1\n",
        config,
    ]
    ran: list[tuple[str, ...]] = []

    def fake(*argv: str) -> str:
        ran.append(argv)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(slurm, "run", fake)

    # No Slurm returns None; scontrol errors fail loudly and are not cached.
    assert slurm.cluster() is None
    with pytest.raises(subprocess.TimeoutExpired):
        slurm.cluster()
    with pytest.raises(ValueError, match="ClusterName"):
        slurm.cluster()
    assert [slurm.cluster() for _ in range(3)] == ["other"] * 3
    assert ran == [("scontrol", "show", "config")] * 4

    slurm._cluster = None
    answers.append(config.replace("other", "third"))
    assert slurm.cluster() == "third"


# --- whether a job is gone ---------------------------------------------------


def test_a_job_the_queue_forgot_and_accounting_ended_cleanly_is_gone(
    answers: Answers,
) -> None:
    ran = answers(squeue=_failed("squeue", FORGOTTEN), sacct="123|COMPLETED|None\n")

    assert slurm.gone("sc", "123", 0)
    assert ran == [SQUEUE, SACCT]


@pytest.mark.parametrize(
    "accounted",
    [
        "123|FAILED|None\n",
        "123|CANCELLED by 24928|None\n",
        "123|TIMEOUT|None\n",
        "123|OUT_OF_MEMORY|None\n",
        "123|DEADLINE|None\n",
        "123|REVOKED|None\n",
        "123|PREEMPTED|None\n",
        # A row for each time a requeued job ran.
        "123|PREEMPTED|None\n123|PREEMPTED|BeginTime\n123|COMPLETED|None\n",
        "123|REQUEUED|None\n123|COMPLETED|None\n",
        "123|FAILED|None\n123|REQUEUED|None\n",
        # The other tasks of its array say nothing about it.
        "122|RUNNING|None\n123|COMPLETED|None\n124|NODE_FAIL|Kill task failed\n",
    ],
)
def test_every_clean_end_is_gone(accounted: str, answers: Answers) -> None:
    # The queue can also answer with nothing for a job it has forgotten.
    answers(squeue="", sacct=accounted)

    assert slurm.gone("sc", "123", 0)


@pytest.mark.parametrize(
    "queued",
    [
        "123 COMPLETED 0\n",
        "123 CANCELLED 0\n",
        "123 PREEMPTED 0\n",
        # The other tasks of its array are listed with it, and may be running.
        "122 RUNNING 0\n123 TIMEOUT 0\n124 PENDING 0\n",
    ],
)
def test_a_job_the_queue_still_remembers_as_ended_is_gone(
    queued: str, answers: Answers
) -> None:
    ran = answers(squeue=queued, sacct="123|COMPLETED|None\n")

    assert slurm.gone("sc", "123", 0)
    assert ran == [SQUEUE, SACCT]


@pytest.mark.parametrize(
    "queued",
    [
        "123 RUNNING 0\n",
        "123 PENDING 0\n",
        "123 COMPLETING 0\n",
        "123 STAGE_OUT 0\n",
        "123 NODE_FAIL 0\n",
        "123 BOOT_FAIL 0\n",
        "123 SOMETHING_NEW 0\n",
        "122 COMPLETED 0\n123 RUNNING 0\n",
    ],
)
def test_a_job_the_queue_lists_as_anything_else_is_not_gone(
    queued: str, answers: Answers
) -> None:
    ran = answers(squeue=queued, sacct="123|COMPLETED|None\n")

    assert not slurm.gone("sc", "123", 0)
    assert ran == [SQUEUE]  # accounting is not asked


@pytest.mark.parametrize(
    ("queued", "accounted", "gone"),
    [
        ("RUNNING 1", "123|REQUEUED|None\n123|RUNNING|None\n", [True, False]),
        ("PENDING 1", "123|PREEMPTED|None\n", [True, False]),
        ("RUNNING 2", "123|PREEMPTED|None\n123|REQUEUED|None\n", [True, True, False]),
        ("RUNNING 2", "123|FAILED|None\n123|REQUEUED|None\n", [True, True, False]),
        # Its processes are still being killed.
        ("COMPLETING 1", "123|REQUEUED|None\n", [False, False]),
        ("REQUEUED 1", "123|REQUEUED|None\n", [False, False]),
        # Accounting is behind the queue.
        ("RUNNING 1", "", [False, False]),
        ("RUNNING 2", "123|REQUEUED|None\n123|RUNNING|None\n", [True, False, False]),
        # A node that stopped answering may still run what it ran.
        ("RUNNING 1", "123|NODE_FAIL|None\n123|RUNNING|None\n", [False, False]),
        ("RUNNING 2", "123|NODE_FAIL|None\n123|PREEMPTED|None\n", [False] * 3),
        ("RUNNING 2", "123|PREEMPTED|None\n123|BOOT_FAIL|None\n", [False] * 3),
        ("RUNNING 1", "123|SOMETHING_NEW|None\n123|REQUEUED|None\n", [False] * 2),
        ("RUNNING 1", "123|PREEMPTED|kill task failed\n", [False, False]),
    ],
)
def test_a_run_of_a_job_the_queue_started_again_is_gone(
    queued: str, accounted: str, gone: list[bool], answers: Answers
) -> None:
    # SLURM_RESTART_COUNT is one higher each time a requeued job runs again.
    answers(squeue=f"122 COMPLETED 0\n123 {queued}\n", sacct=accounted)

    assert [slurm.gone("sc", "123", restart) for restart in range(len(gone))] == gone


def test_a_job_that_was_not_started_again_is_not_asked_of_accounting(
    answers: Answers,
) -> None:
    ran = answers(squeue="123 RUNNING 1\n", sacct="123|REQUEUED|None\n")

    assert not slurm.gone("sc", "123", 1)
    assert ran == [SQUEUE]


@pytest.mark.parametrize(
    "accounted",
    [
        "",  # accounting is behind the queue
        "123|NODE_FAIL|None\n",
        "123|BOOT_FAIL|None\n",
        "123|COMPLETED|Kill task failed\n",
        "123|FAILED|kill task failed on node jagupard26\n",
        "123|RUNNING|None\n",
        "123|PENDING|None\n",
        # Requeued, and the row of the run that followed is not there yet.
        "123|REQUEUED|None\n",
        "123|REQUEUED|None\n123|RUNNING|None\n",
        "123|NODE_FAIL|None\n123|COMPLETED|None\n",
        "123|SOMETHING_NEW|None\n",
        "124|COMPLETED|None\n",  # another job's row only
    ],
)
def test_an_end_accounting_does_not_call_clean_is_not_gone(
    accounted: str, answers: Answers
) -> None:
    answers(squeue=_failed("squeue", FORGOTTEN), sacct=accounted)

    assert not slurm.gone("sc", "123", 0)


@pytest.mark.parametrize(
    ("table", "error"),
    [
        # The queue fails for another reason than a forgotten job.
        (
            {
                "squeue": _failed("squeue", "slurm_load_jobs error: Socket timed out"),
                "sacct": "123|COMPLETED|None\n",
            },
            subprocess.CalledProcessError,
        ),
        # It printed something before it failed.
        (
            {
                "squeue": _failed("squeue", FORGOTTEN, stdout="123 RUNNING 0\n"),
                "sacct": "123|COMPLETED|None\n",
            },
            subprocess.CalledProcessError,
        ),
        (
            {
                "squeue": subprocess.TimeoutExpired("squeue", 60.0),
                "sacct": "123|COMPLETED|None\n",
            },
            subprocess.TimeoutExpired,
        ),
        (
            {"squeue": "", "sacct": _failed("sacct", "sacct: error")},
            subprocess.CalledProcessError,
        ),
        (
            {"squeue": "", "sacct": subprocess.TimeoutExpired("sacct", 60.0)},
            subprocess.TimeoutExpired,
        ),
        ({"squeue": ""}, FileNotFoundError),  # a Slurm machine without sacct
        ({}, FileNotFoundError),  # a Slurm machine without squeue
        ({"squeue": "123 RUNNING\n", "sacct": "123|COMPLETED|None\n"}, ValueError),
        (
            {"squeue": "123 RUNNING 0 extra\n", "sacct": "123|COMPLETED|None\n"},
            ValueError,
        ),
        ({"squeue": "123 RUNNING N/A\n", "sacct": "123|COMPLETED|None\n"}, ValueError),
        ({"squeue": "", "sacct": "123|COMPLETED\n"}, ValueError),
        (
            {"squeue": "", "sacct": "123|COMPLETED|None\nnot a row at all\n"},
            ValueError,
        ),
    ],
)
def test_a_slurm_failure_or_unexpected_output_raises(
    table: dict[str, str | Exception], error: type[Exception], answers: Answers
) -> None:
    answers(**table)

    with pytest.raises(error):
        slurm.gone("sc", "123", 0)


@pytest.mark.parametrize("cluster", ["other", None])
def test_a_job_of_another_cluster_is_not_gone_and_slurm_is_not_asked(
    cluster: str | None, answers: Answers
) -> None:
    ran = answers(cluster, squeue="", sacct="123|COMPLETED|None\n")

    assert not slurm.gone("sc", "123", 0)
    assert ran == []


def test_the_cluster_a_job_is_checked_against_is_asked_of_the_controller_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[str] = []

    def fake(*argv: str) -> str:
        ran.append(argv[0])
        return {
            "scontrol": "ClusterName = sc\n",
            "squeue": "",
            "sacct": "123|COMPLETED|None\n",
        }[argv[0]]

    monkeypatch.setattr(slurm, "run", fake)

    assert slurm.gone("sc", "123", 0) and slurm.gone("sc", "123", 0)
    assert not slurm.gone("other", "123", 0)
    assert ran == ["scontrol", "squeue", "sacct", "squeue", "sacct"]


# --- the steps of a job ------------------------------------------------------

STEP = ["srun", "--exact", "--ntasks=1"]
NAMED = ["--export=ALL", "--job-name=furu-a"]
STEPS = (
    *("squeue", "--steps", "--noheader", "--format=%i %j"),
    *("-j", "100"),
)


@pytest.fixture
def job(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """This process as a Slurm job with what the keywords say, in the words
    of Slurm's variables: ``cpus_on_node=8`` is ``SLURM_CPUS_ON_NODE=8``."""

    def install(threads_per_core: int = 1, **variables: int) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "100")
        for name, value in variables.items():
            monkeypatch.setenv(f"SLURM_{name.upper()}", str(value))
        monkeypatch.setattr(slurm, "_threads_per_core", lambda: threads_per_core)

    return install


def test_a_step_asks_for_its_share_of_what_the_job_has(
    job: Callable[..., None],
) -> None:
    job(cpus_on_node=64, gpus_on_node=8, mem_per_node=512 * 1024)

    assert [slurm.step("furu-a", taken=taken, slots=8) for taken in (1, 2, 4, 8)] == [
        [*STEP, "--cpus-per-task=8", "--gpus=1", "--mem=65536M", *NAMED],
        [*STEP, "--cpus-per-task=16", "--gpus=2", "--mem=131072M", *NAMED],
        [*STEP, "--cpus-per-task=32", "--gpus=4", "--mem=262144M", *NAMED],
        [*STEP, "--cpus-per-task=64", "--gpus=8", "--mem=524288M", *NAMED],
    ]


def test_a_step_of_a_job_without_gpus_or_a_memory_limit_asks_for_neither(
    job: Callable[..., None],
) -> None:
    job(cpus_on_node=6)

    # A share is rounded down, so that every slot has one.
    assert slurm.step("furu-a", taken=1, slots=4) == [
        *STEP,
        "--cpus-per-task=1",
        *NAMED,
    ]
    assert slurm.step("furu-a", taken=2, slots=4)[3] == "--cpus-per-task=3"


@pytest.mark.parametrize(
    ("memory", "half"),
    [
        # By the CPU or the GPU, a job has what all of them come to.
        ({"mem_per_cpu": 1024}, "--mem=4096M"),
        ({"mem_per_gpu": 3000}, "--mem=3000M"),
        ({"mem_per_node": 5000}, "--mem=2500M"),
        # Slurm sets a second one that says nothing of the job.
        ({"mem_per_cpu": 1024, "mem_per_node": 49152}, "--mem=4096M"),
        ({"mem_per_gpu": 3000, "mem_per_node": 49152}, "--mem=3000M"),
    ],
)
def test_a_step_always_says_how_much_memory_it_takes(
    memory: dict[str, int], half: str, job: Callable[..., None]
) -> None:
    job(cpus_on_node=8, gpus_on_node=2, **memory)

    assert slurm.step("furu-a", taken=1, slots=2) == [
        *STEP,
        "--cpus-per-task=4",
        "--gpus=1",
        half,
        *NAMED,
    ]


def test_a_step_takes_whole_cores_where_two_cpus_share_one(
    job: Callable[..., None],
) -> None:
    # 5 cores: a slot of 3 CPUs would take 2 of them, and 4 slots would wait
    # for each other.
    job(threads_per_core=2, cpus_on_node=10)

    cpus = [slurm.step("furu-a", taken=taken, slots=4)[3] for taken in (1, 2, 4)]

    assert cpus == ["--cpus-per-task=2", "--cpus-per-task=4", "--cpus-per-task=10"]


def test_a_job_with_fewer_cores_than_slots_cannot_be_cut_into_steps(
    job: Callable[..., None],
) -> None:
    job(threads_per_core=2, cpus_on_node=4)

    with pytest.raises(RuntimeError, match="2 cores for 4 slots.*at least 8 CPUs"):
        slurm.step("furu-a", taken=1, slots=4)
    assert slurm.step("furu-a", taken=2, slots=4)[3] == "--cpus-per-task=2"


def test_the_cpus_this_process_has_share_cores_evenly() -> None:
    threads = slurm._threads_per_core()

    assert threads >= 1
    assert len(os.sched_getaffinity(0)) % threads == 0


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slurm.time, "sleep", lambda seconds: None)


@pytest.fixture
def steps(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> Callable[..., list[tuple[str, ...]]]:
    """Answer squeue with each listing in turn, the last from then on, and
    scancel with ``scancel``. Returns the commands that were run."""

    def install(*listings: str, scancel: str | Exception = "") -> list[tuple[str, ...]]:
        ran: list[tuple[str, ...]] = []
        left = list(listings)

        def fake(*argv: str) -> str:
            ran.append(argv)
            if argv[0] == "squeue":
                return left.pop(0) if len(left) > 1 else left[0]
            if isinstance(scancel, Exception):
                raise scancel
            return scancel

        monkeypatch.setenv("SLURM_JOB_ID", "100")
        monkeypatch.setattr(slurm, "run", fake)
        return ran

    return install


OTHERS = "100.batch batch\n100.extern extern\n100.4 furu-b\n"


def test_a_step_slurm_no_longer_lists_is_ended_without_a_kill(
    steps: Callable[..., list[tuple[str, ...]]],
) -> None:
    ran = steps(OTHERS)

    slurm.end_step("furu-a")

    assert ran == [STEPS]


def test_a_step_is_killed_until_slurm_no_longer_lists_it(
    steps: Callable[..., list[tuple[str, ...]]],
) -> None:
    # An array task's steps are named for the array.
    listed = OTHERS + "100_3.7 furu-a\n"
    ran = steps(listed, listed, OTHERS)

    slurm.end_step("furu-a")

    kill = ("scancel", "--signal=KILL", "100_3.7")
    assert ran == [STEPS, kill, STEPS, kill, STEPS]


def test_a_kill_that_fails_is_tried_again(
    steps: Callable[..., list[tuple[str, ...]]],
) -> None:
    # As when the step ended between the listing and the kill.
    ran = steps("100.7 furu-a\n", OTHERS, scancel=_failed("scancel", "Invalid job id"))

    slurm.end_step("furu-a")

    assert [argv[0] for argv in ran] == ["squeue", "scancel", "squeue"]


def test_a_step_that_outlives_its_kill_is_an_error(
    steps: Callable[..., list[tuple[str, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    steps("100.7 furu-a\n")
    monkeypatch.setattr(slurm, "_STEP_END_TIMEOUT_S", -1.0)

    with pytest.raises(TimeoutError, match=r"still lists the killed steps \['100.7'\]"):
        slurm.end_step("furu-a")


def test_a_queue_that_does_not_answer_ends_no_step(
    monkeypatch: pytest.MonkeyPatch, answers: Answers
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    answers(squeue=_failed("squeue", "slurm_load_jobs error: Socket timed out"))

    with pytest.raises(subprocess.CalledProcessError):
        slurm.end_step("furu-a")
