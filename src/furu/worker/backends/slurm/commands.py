import os
import re
import subprocess


def slurm(*argv: str) -> str:
    """Run a Slurm command and return its output.

    Raises CalledProcessError if it fails and TimeoutExpired if it hangs.
    """
    # User display filters (e.g. SQUEUE_PARTITION) can hide a live owner.
    env = None
    if argv[0] in ("squeue", "sacct"):
        prefix = argv[0].upper() + "_"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(prefix)
        }
    return subprocess.run(
        argv,
        check=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def clean_exit(cluster: str, job: str, restart: str) -> bool:
    """Prove the allocation is gone and accounting covers this incarnation."""
    if (
        os.environ.get("SLURM_CLUSTER_NAME") != cluster
        or os.environ.get("SLURM_JOB_ID") == job
    ):
        return False
    try:
        # Query all states, including COMPLETING and hidden partitions. A
        # filtered -j query can fail once the controller forgets an old job.
        queue = slurm(
            "squeue",
            "-M",
            cluster,
            "--all",
            "--array",
            "--states=all",
            "-h",
            "-o",
            "%A",
        ).splitlines()
        if job in queue or any(not re.fullmatch(r"[0-9]+", line) for line in queue):
            return False
        rows = slurm(
            "sacct",
            "-M",
            cluster,
            "-D",
            "-X",
            "-nP",
            "-j",
            job,
            "-o",
            "JobIDRaw,Restarts,State,Reason,SystemComment",
        ).splitlines()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return False
    covered = False
    for row in rows:
        fields = row.split("|")
        if len(fields) != 5:
            return False
        raw_id, count, state, reason, comment = fields
        if (
            raw_id != job
            or not re.fullmatch(r"[0-9]+", count)
            or not re.fullmatch(
                r"COMPLETED|CANCELLED(?: by [0-9]+)?|FAILED|TIMEOUT|OUT_OF_MEMORY|DEADLINE",
                state,
            )
            or "kill task failed" in (reason + " " + comment).lower()
        ):
            return False
        covered |= count == restart
    return covered
