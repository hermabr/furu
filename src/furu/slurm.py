"""Running Slurm's commands, and asking Slurm whether a job is gone."""

from __future__ import annotations

import os
import subprocess

# The states of a job whose processes Slurm saw end. Not NODE_FAIL or
# BOOT_FAIL: a node that stopped answering can still be running them.
_GONE = frozenset(
    {
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

# The cluster the controller named, once it has: asked at most once a process.
_cluster: list[str] = []


def run(*argv: str) -> str:
    """Run a Slurm command and return its output.

    Raises CalledProcessError if it fails, TimeoutExpired if it hangs and
    FileNotFoundError on a machine without Slurm. The variables that change
    what squeue and sacct print, or which cluster a command asks, are not
    passed on: the output is read by code.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("SQUEUE_", "SACCT_")) and name != "SLURM_CLUSTERS"
    }
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=60.0,
        env=environment,
    ).stdout


def cluster() -> str | None:
    """The Slurm cluster this machine belongs to, if it can be told.

    A job has it in its environment; any other machine asks the controller.
    """
    if name := os.environ.get("SLURM_CLUSTER_NAME"):
        return name
    if not _cluster:
        try:
            config = run("scontrol", "show", "config")
        except (OSError, subprocess.SubprocessError):
            return None
        for line in config.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ClusterName" and value.strip():
                _cluster.append(value.strip())
                break
    return _cluster[0] if _cluster else None


def forget_cluster() -> None:
    """Ask the controller again the next time: for tests."""
    _cluster.clear()


def gone(cluster_name: str, job: str) -> bool:
    """Whether Slurm says that ``job`` ended and left no process behind.

    True only when the queue no longer runs the job and accounting has a clean
    end for it. A command that fails, an answer that is not understood and a
    job of another cluster are all False: the job may be alive.
    """
    if cluster_name != cluster():
        return False
    # Asked by id, the queue also lists a job that ended, for as long as the
    # controller remembers it. Not -O JobIDRaw, which Slurm 25.11.5 refuses.
    try:
        queued = run(
            "squeue", "--all", "--array", "--noheader", "--format=%A %T", "-j", job
        )
    except subprocess.CalledProcessError as exc:
        # How the queue answers for a job it no longer remembers.
        if (exc.stdout or "").strip() or "Invalid job id" not in (exc.stderr or ""):
            return False
        queued = ""
    except (OSError, subprocess.SubprocessError):
        return False
    for line in queued.splitlines():
        match line.split():
            case [job_id, state]:
                # The other tasks of its array are listed with it.
                if job_id == job and state not in _GONE:
                    return False
            case _:
                return False
    # -D: a requeued job has a row for each time it ran.
    try:
        accounted = run(
            *("sacct", "-D", "-X", "--noheader", "--parsable2"),
            *("-o", "JobIDRaw,State,Reason", "-j", job),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    states = []
    for line in accounted.splitlines():
        match line.split("|", 2):
            case [raw_id, state, reason]:
                if raw_id != job:
                    continue
                # Accounting ended the job while a process could not be killed.
                if "kill task failed" in reason.lower():
                    return False
                states.append(state.partition(" ")[0])  # "CANCELLED by <uid>"
            case _:
                return False
    # A row that says the job was requeued is no end: accounting can have it
    # before the row of the run that followed. Without any row it is behind
    # the queue.
    return all(state in _GONE | {"REQUEUED"} for state in states) and any(
        state in _GONE for state in states
    )
