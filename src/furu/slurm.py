"""Slurm's commands, run one way for the worker pools and the attempt locks."""

from __future__ import annotations

import os
import subprocess

TIMEOUT_S = 60.0

# How a job may have ended for none of its processes to run any more.
# NODE_FAIL and BOOT_FAIL are not here: the node may still run them.
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

# The ClusterName of the Slurm config, once scontrol has given it.
_cluster: str | None = None


def run(*argv: str) -> str:
    """Run a Slurm command and return its output.

    The user's SQUEUE_*, SACCT_* and SLURM_CLUSTERS settings are left out, so
    the output has the format asked for, from this cluster. Raises
    CalledProcessError if it fails, TimeoutExpired if it hangs and
    FileNotFoundError where Slurm is not installed.
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
        timeout=TIMEOUT_S,
        env=environment,
    ).stdout


def cluster() -> str | None:
    """The Slurm cluster this process runs on, if it can tell."""
    global _cluster
    if name := os.environ.get("SLURM_CLUSTER_NAME"):
        return name
    if _cluster is None:
        try:
            config = run("scontrol", "show", "config")
        except (OSError, subprocess.SubprocessError):
            return None  # asked again next time: Slurm may just be slow
        for line in config.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ClusterName" and value.strip():
                _cluster = value.strip()
    return _cluster


def gone(cluster_name: str, job: str) -> bool:
    """Whether Slurm job ``job`` of ``cluster_name`` surely runs no more.

    Only when squeue lists it as ended or not at all, and accounting has it
    ended in a way that leaves no process behind. Any doubt is False.
    """
    if cluster_name != cluster():
        return False
    try:
        queued = run(
            "squeue", "--all", "--array", "--noheader", "--format=%A %T", "-j", job
        )
    except subprocess.CalledProcessError as exc:
        # Older Slurm says so of a job it has purged; newer lists nothing.
        if (exc.stdout or "").strip() or "Invalid job id" not in (exc.stderr or ""):
            return False
        queued = ""
    except (OSError, subprocess.SubprocessError):
        return False
    for line in queued.splitlines():
        job_id, _, state = line.strip().partition(" ")
        if job_id == job and state.strip() not in _GONE:
            return False
    try:
        accounted = run(
            "sacct", "-D", "-X", "--noheader", "--parsable2",
            "-o", "JobIDRaw,State,Reason", "-j", job,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return False
    # -D lists each run of a requeued job: every one must have ended, and the
    # last with a final state, which accounting may not have yet.
    ends = []
    for line in accounted.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 2)
        if len(fields) != 3:
            return False
        raw_id, state, reason = fields
        if raw_id != job:
            continue  # another task of its array
        state = state.partition(" ")[0]  # "CANCELLED by <uid>" is CANCELLED
        if state not in _GONE | {"REQUEUED"} or "kill task failed" in reason.lower():
            return False
        ends.append(state)
    return any(state in _GONE for state in ends)
