"""Slurm's commands, run one way for the worker pools and the attempt locks."""

from __future__ import annotations

import os
import subprocess
from functools import cache

TIMEOUT_S = 60.0


def run(*argv: str) -> str:
    """Run a Slurm command and return its output.

    Raises CalledProcessError if it fails, TimeoutExpired if it hangs and
    FileNotFoundError where Slurm is not installed.
    """
    return subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=TIMEOUT_S
    ).stdout


def cluster() -> str | None:
    """The Slurm cluster this process runs on, if it can tell."""
    return os.environ.get("SLURM_CLUSTER_NAME") or _configured_cluster()


@cache
def _configured_cluster() -> str | None:
    """ClusterName in the Slurm config, for a process outside any job."""
    try:
        config = run("scontrol", "show", "config")
    except (OSError, subprocess.SubprocessError):
        return None
    for line in config.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ClusterName":
            return value.strip() or None
    return None
