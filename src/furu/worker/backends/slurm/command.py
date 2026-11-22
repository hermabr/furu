"""Running Slurm's commands: the one place furu starts them."""

from __future__ import annotations

import os
import subprocess

_SLURM_COMMAND_TIMEOUT_S = 60.0


def slurm(*argv: str) -> str:
    """Run a Slurm command and return its output.

    Raises CalledProcessError if it fails, TimeoutExpired if it hangs and
    FileNotFoundError on a machine without Slurm.
    """
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=_SLURM_COMMAND_TIMEOUT_S,
    ).stdout


def cluster() -> str | None:
    """The Slurm cluster this machine belongs to, if it can be told.

    A job has it in its environment; a login node asks the controller.
    """
    if name := os.environ.get("SLURM_CLUSTER_NAME"):
        return name
    try:
        config = slurm("scontrol", "show", "config")
    except (OSError, subprocess.SubprocessError):
        return None
    for line in config.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ClusterName":
            return value.strip() or None
    return None
