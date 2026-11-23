"""How Slurm's commands are run, and what Slurm has to say before a job counts
as gone."""

import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from furu import slurm

FORGOTTEN = "slurm_load_jobs error: Invalid job id specified\n"
SQUEUE = (
    *("squeue", "--all", "--array", "--noheader", "--format=%A %T"),
    *("-j", "123"),
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

    # No answer is not remembered: the controller may answer the next time.
    assert [slurm.cluster() for _ in range(6)] == [None, None, None, *["other"] * 3]
    assert ran == [("scontrol", "show", "config")] * 4

    monkeypatch.setattr(slurm, "_cluster", None)
    answers.append(config.replace("other", "third"))
    assert slurm.cluster() == "third"


# --- whether a job is gone ---------------------------------------------------


def test_a_job_the_queue_forgot_and_accounting_ended_cleanly_is_gone(
    answers: Answers,
) -> None:
    ran = answers(squeue=_failed("squeue", FORGOTTEN), sacct="123|COMPLETED|None\n")

    assert slurm.gone("sc", "123")
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

    assert slurm.gone("sc", "123")


@pytest.mark.parametrize(
    "queued",
    [
        "123 COMPLETED\n",
        "123 CANCELLED\n",
        "123 PREEMPTED\n",
        # The other tasks of its array are listed with it, and may be running.
        "122 RUNNING\n123 TIMEOUT\n124 PENDING\n",
    ],
)
def test_a_job_the_queue_still_remembers_as_ended_is_gone(
    queued: str, answers: Answers
) -> None:
    ran = answers(squeue=queued, sacct="123|COMPLETED|None\n")

    assert slurm.gone("sc", "123")
    assert ran == [SQUEUE, SACCT]


@pytest.mark.parametrize(
    "queued",
    [
        "123 RUNNING\n",
        "123 PENDING\n",
        "123 COMPLETING\n",
        "123 STAGE_OUT\n",
        "123 NODE_FAIL\n",
        "123 BOOT_FAIL\n",
        "123 SOMETHING_NEW\n",
        "122 COMPLETED\n123 RUNNING\n",
        "123\n",
        "123 RUNNING extra\n",
    ],
)
def test_a_job_the_queue_lists_as_anything_else_is_not_gone(
    queued: str, answers: Answers
) -> None:
    ran = answers(squeue=queued, sacct="123|COMPLETED|None\n")

    assert not slurm.gone("sc", "123")
    assert ran == [SQUEUE]  # accounting is not asked


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
        "123|COMPLETED\n",
        "123|COMPLETED|None\nnot a row at all\n",
    ],
)
def test_an_end_accounting_does_not_call_clean_is_not_gone(
    accounted: str, answers: Answers
) -> None:
    answers(squeue=_failed("squeue", FORGOTTEN), sacct=accounted)

    assert not slurm.gone("sc", "123")


@pytest.mark.parametrize(
    "table",
    [
        # The queue fails for another reason than a forgotten job.
        {
            "squeue": _failed("squeue", "slurm_load_jobs error: Socket timed out"),
            "sacct": "123|COMPLETED|None\n",
        },
        # It printed something before it failed.
        {
            "squeue": _failed("squeue", FORGOTTEN, stdout="123 RUNNING\n"),
            "sacct": "123|COMPLETED|None\n",
        },
        {
            "squeue": subprocess.TimeoutExpired("squeue", 60.0),
            "sacct": "123|COMPLETED|None\n",
        },
        {"squeue": "", "sacct": _failed("sacct", "sacct: error")},
        {"squeue": "", "sacct": subprocess.TimeoutExpired("sacct", 60.0)},
        {"squeue": ""},  # a machine without sacct
        {},  # a machine without Slurm
    ],
)
def test_a_job_slurm_does_not_answer_for_is_not_gone(
    table: dict[str, str | Exception], answers: Answers
) -> None:
    answers(**table)

    assert not slurm.gone("sc", "123")


@pytest.mark.parametrize("cluster", ["other", None])
def test_a_job_of_another_cluster_is_not_gone_and_slurm_is_not_asked(
    cluster: str | None, answers: Answers
) -> None:
    ran = answers(cluster, squeue="", sacct="123|COMPLETED|None\n")

    assert not slurm.gone("sc", "123")
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

    assert slurm.gone("sc", "123") and slurm.gone("sc", "123")
    assert not slurm.gone("other", "123")
    assert ran == ["scontrol", "squeue", "sacct", "squeue", "sacct"]
