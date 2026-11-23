"""Shared Slurm command runner and conservative job-termination check."""

from __future__ import annotations

import os
import subprocess

_SLURM_COMMAND_TIMEOUT_S = 60.0

# Terminal states that guarantee no task of the job can still be running.
# NODE_FAIL and BOOT_FAIL stay excluded: a partitioned node may still hold a
# zombie process writing to shared storage.
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

_cluster_cache: str | None = None


def clear_cluster_cache() -> None:
    """Reset the cached ``scontrol show config`` cluster name (for tests)."""
    global _cluster_cache
    _cluster_cache = None


def run(*argv: str) -> str:
    """Run a Slurm command with formatting/cluster env vars scrubbed."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SQUEUE_", "SACCT_")) and key != "SLURM_CLUSTERS"
    }
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=_SLURM_COMMAND_TIMEOUT_S,
        env=env,
    ).stdout


def cluster() -> str | None:
    """This Slurm cluster's name, or None when Slurm is not reachable."""
    global _cluster_cache
    if name := os.environ.get("SLURM_CLUSTER_NAME"):
        return name
    if _cluster_cache is not None:
        return _cluster_cache
    try:
        output = run("scontrol", "show", "config")
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "ClusterName" and (name := value.strip()):
            _cluster_cache = name
            return name
    return None


def gone(cluster_name: str, job: str) -> bool:
    """Whether Slurm positively confirms ``job`` on ``cluster_name`` is done.

    Strictly fail-closed: returns False on any error, ambiguity, or mismatch.
    """
    if cluster_name != cluster():
        return False

    # Gate 1: squeue must not report this exact job in any non-terminal state.
    try:
        out = run(
            "squeue",
            "--all",
            "--array",
            "--noheader",
            "--format=%A %T",
            "-j",
            job,
        )
    except subprocess.CalledProcessError as exc:
        if (exc.stdout or "").strip() or "Invalid job id" not in (exc.stderr or ""):
            return False
        out = ""
    except (OSError, subprocess.SubprocessError):
        return False

    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 2:
            return False
        if parts[0] == job and parts[1] not in _GONE:
            return False

    # Gate 2: sacct must affirmatively record a clean terminal row for this job.
    try:
        out = run(
            "sacct",
            "-D",
            "-X",
            "--noheader",
            "--parsable2",
            "-o",
            "JobIDRaw,State,Reason",
            "-j",
            job,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            return False
        raw_id, raw_state, reason = parts
        if raw_id != job:
            continue
        rows.append((raw_state.strip().partition(" ")[0], reason))

    if not rows:
        return False
    allowed = _GONE | {"REQUEUED"}
    return all(
        state in allowed and "kill task failed" not in reason.lower()
        for state, reason in rows
    ) and any(state in _GONE for state, _ in rows)
