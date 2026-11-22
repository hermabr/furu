from __future__ import annotations

import subprocess

SLURM_COMMAND_TIMEOUT_S = 60.0


def run_slurm(*argv: str) -> str:
    """Run a Slurm command and return its stdout.

    Raises CalledProcessError if it fails, TimeoutExpired if it hangs, and
    OSError (e.g. FileNotFoundError) if the binary is unavailable.
    """
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=SLURM_COMMAND_TIMEOUT_S,
    ).stdout
