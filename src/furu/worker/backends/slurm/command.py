import subprocess


def slurm(*argv: str) -> str:
    """Run a Slurm command and return its output.

    Raises CalledProcessError if it fails and TimeoutExpired if it hangs.
    """
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
