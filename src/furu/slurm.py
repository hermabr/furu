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


class SlurmError(RuntimeError):
    """A Slurm command failed, timed out, or returned unexpected output."""


def _error_detail(exc: BaseException) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        detail = (exc.stderr or exc.stdout or "").strip()
        return detail or f"exit status {exc.returncode}"
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"timed out after {exc.timeout:g}s"
    return str(exc)


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
    """The Slurm cluster this machine belongs to, or None if Slurm is not installed.

    A job has it in its environment; any other machine asks the controller.
    Raises SlurmError if ``scontrol`` is installed but fails or omits ClusterName.
    """
    if name := os.environ.get("SLURM_CLUSTER_NAME"):
        return name
    if not _cluster:
        try:
            config = run("scontrol", "show", "config")
        except FileNotFoundError:
            return None
        except (OSError, subprocess.SubprocessError) as exc:
            raise SlurmError(
                f"scontrol show config failed: {_error_detail(exc)}"
            ) from exc
        for line in config.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ClusterName" and value.strip():
                _cluster.append(value.strip())
                break
        else:
            raise SlurmError("scontrol show config did not report ClusterName")
    return _cluster[0]


def forget_cluster() -> None:
    """Ask the controller again the next time: for tests."""
    _cluster.clear()


def why_not_gone(cluster_name: str, job: str) -> str | None:
    """Why Slurm does not report ``job`` as cleanly ended; None if it is gone.

    Returns None only when the queue no longer runs the job and accounting has
    a clean end for it. Returns a human-readable reason when the job may still
    be alive or belongs to another cluster. Raises SlurmError when a Slurm
    command fails, times out, or returns unparseable output.
    """
    local = cluster()
    if local is None:
        return "this machine is not on a Slurm cluster"
    if cluster_name != local:
        return f"this machine is on Slurm cluster {local}, not {cluster_name}"
    # Asked by id, the queue also lists a job that ended, for as long as the
    # controller remembers it. Not -O JobIDRaw, which Slurm 25.11.5 refuses.
    try:
        queued = run(
            "squeue", "--all", "--array", "--noheader", "--format=%A %T", "-j", job
        )
    except subprocess.CalledProcessError as exc:
        # How the queue answers for a job it no longer remembers.
        if (exc.stdout or "").strip() or "Invalid job id" not in (exc.stderr or ""):
            raise SlurmError(
                f"squeue failed for job {job}: {_error_detail(exc)}"
            ) from exc
        queued = ""
    except (OSError, subprocess.SubprocessError) as exc:
        raise SlurmError(f"squeue failed for job {job}: {_error_detail(exc)}") from exc
    for line in queued.splitlines():
        match line.split():
            case [job_id, state]:
                # The other tasks of its array are listed with it.
                if job_id == job and state not in _GONE:
                    return f"squeue reports state {state}"
            case _:
                raise SlurmError(
                    f"squeue returned unexpected output for job {job}: {line!r}"
                )
    # -D: a requeued job has a row for each time it ran.
    try:
        accounted = run(
            *("sacct", "-D", "-X", "--noheader", "--parsable2"),
            *("-o", "JobIDRaw,State,Reason", "-j", job),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SlurmError(f"sacct failed for job {job}: {_error_detail(exc)}") from exc
    states: list[str] = []
    for line in accounted.splitlines():
        match line.split("|", 2):
            case [raw_id, state, reason]:
                if raw_id != job:
                    continue
                # Accounting ended the job while a process could not be killed.
                if "kill task failed" in reason.lower():
                    return f"sacct reports {reason.strip()}"
                states.append(state.partition(" ")[0])  # "CANCELLED by <uid>"
            case _:
                raise SlurmError(
                    f"sacct returned unexpected output for job {job}: {line!r}"
                )
    if not states:
        return "sacct has no accounting record for it yet"
    # A row that says the job was requeued is no end: accounting can have it
    # before the row of the run that followed.
    if all(state in _GONE | {"REQUEUED"} for state in states) and any(
        state in _GONE for state in states
    ):
        return None
    label = "state" if len(states) == 1 else "states"
    return f"sacct reports {label} {', '.join(states)}"
