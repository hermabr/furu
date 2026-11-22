"""Who works on a run: the name of the one working folder in its attempts/<run>/.

attempts/<run>/ holds at most one working folder, named for its owner:

    free__<tok>                                   nobody; anyone may claim it
    slurm__<cluster>__<job id>__<restart>__<tok>  a process of that Slurm job
    local__<host>__<pid>__<tok>                   that process, outside Slurm

A claim renames the folder to the claimer's name with a fresh token, and a
release renames it to ``free__<fresh tok>``. A rename of one source has one
winner over NFS, and a fresh token means a late retry of an old rename can
never match a new name. An empty attempts/<run>/ is seeded with ``free__seed``.
Names that start with a dot are trash on its way out and are ignored.

Liveness comes from Slurm and never from a clock: a folder of a Slurm job is
taken over once Slurm says that every incarnation of the job ended cleanly. A
``local__`` folder is never taken over; a killed process strands it, and the
error that finds it says how to free it.
"""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

from furu.logging import get_logger
from furu.storage._layout import log_path_in

SEED = "free__seed"
# Every incarnation of a Slurm job must have ended in one of these before its
# folder is taken over. NODE_FAIL, BOOT_FAIL and an unkillable step can leave
# its processes running; an unknown state might too.
_CLEAN_ENDS = frozenset(
    {
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REQUEUED",
        "TIMEOUT",
    }
)
_WAIT_S = 0.05


def owner(pid: int | None = None) -> str:
    """The name prefix of a folder this process, or its child ``pid``, works in."""
    env = os.environ
    if "SLURM_JOB_ID" in env:
        cluster, job = env["SLURM_CLUSTER_NAME"], env["SLURM_JOB_ID"]
        return f"slurm__{cluster}__{job}__{env.get('SLURM_RESTART_COUNT', '0')}"
    return f"local__{socket.gethostname()}__{pid or os.getpid()}"


def _tok() -> str:
    return secrets.token_hex(8)


def entries(attempt_dir: Path) -> list[str]:
    """The folders in attempts/<run>/ as a listing shows now, trash left out.

    Listing also makes this machine look again at the names in it, which NFS
    otherwise remembers as missing for up to a minute.
    """
    return [name for name in os.listdir(attempt_dir) if not name.startswith(".")]


def working_folder_in(attempt_dir: Path) -> Path | None:
    """The working folder in attempts/<run>/ now, if it has exactly one."""
    try:
        names = entries(attempt_dir)
    except FileNotFoundError:
        return None
    return attempt_dir / names[0] if len(names) == 1 else None


def log_in(attempt_dir: Path) -> Path:
    """The log of the working folder in attempts/<run>/ now; the folder itself
    when it has none yet."""
    folder = working_folder_in(attempt_dir)
    return attempt_dir if folder is None else log_path_in(folder)


def _slurm_ended(cluster: str, job: str) -> bool:
    """Whether Slurm says every incarnation of ``job`` ended and none can still run.

    Fails closed: anything Slurm does not answer plainly counts as running.
    """

    def run(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=60
        )

    try:
        queued = run("squeue", "--noheader", "--jobs", job, "--format=%i")
        if queued.returncode != 0 and "Invalid job id" not in queued.stderr:
            return False  # a purged job is "invalid"; any other error says nothing
        if queued.stdout.strip():
            return False
        rows = run(
            *("sacct", "-D", "-X", "--noheader", "--parsable2"),
            *("-o", "Cluster,State,Reason", "-j", job),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    lines = [line.split("|") for line in rows.stdout.splitlines()]
    return (
        rows.returncode == 0
        and bool(lines)
        and all(
            len(line) == 3
            and line[0] == cluster
            # "CANCELLED by <uid>" is CANCELLED.
            and line[1].partition(" ")[0] in _CLEAN_ENDS
            and "Kill task failed" not in line[2]
            for line in lines
        )
    )


def why_held(name: str) -> str | None:
    """Who holds a working folder of this name; None when it may be claimed."""
    kind, *fields = name.split("__")
    if kind == "free" and len(fields) == 1:
        return None
    if kind == "slurm" and len(fields) == 4:
        cluster, job, restart, _ = fields
        if _slurm_ended(cluster, job):
            return None
        return (
            f"Slurm job {job} (restart {restart}) on cluster {cluster}, "
            "which has not ended cleanly"
        )
    if kind == "local" and len(fields) >= 3:
        *host, pid, _ = fields
        return f"process {pid} on {'__'.join(host)}, which only it releases"
    return "an owner furu does not recognize"


def _maybe_alive_here(name: str) -> bool:
    """Whether the folder may belong to a running process beside this one: one
    of this Slurm job, or one that runs on this machine outside Slurm."""
    if name.startswith(f"{owner()}__") and owner().startswith("slurm__"):
        return True
    prefix = f"local__{socket.gethostname()}__"
    if not name.startswith(prefix):
        return False
    pid = name.removeprefix(prefix).partition("__")[0]
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (ValueError, ProcessLookupError):
        return False
    return True


def refusal(attempt_dir: Path, names: list[str]) -> str:
    """Why attempts/<run>/ cannot be claimed, and how to free it."""
    if len(names) > 1:
        return (
            f"{attempt_dir} holds {len(names)} working folders ({', '.join(names)}), "
            "and a run has at most one. Keep the one to go on from, rename it "
            "to free__released, and delete the others."
        )
    (name,) = names
    return (
        f"{attempt_dir / name} is held by {why_held(name)}. If no process works "
        f"in it, release it: mv {attempt_dir / name} {attempt_dir / 'free__released'}"
    )


def claim(attempt_dir: Path) -> Path:
    """Take the working folder of attempts/<run>/, seeding it when there is none.

    Waits while a process that may be alive beside this one holds it; any
    other holder makes this raise, without making a second folder.
    """
    waited = False
    while True:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        try:
            names = entries(attempt_dir)
            if not names:
                with suppress(FileExistsError):
                    (attempt_dir / SEED).mkdir()
                continue
        except FileNotFoundError:  # a publisher just removed the empty folder
            continue
        if len(names) == 1 and _maybe_alive_here(names[0]):
            if not waited:
                folder = attempt_dir / names[0]
                get_logger().info(
                    "waiting for %s, held beside this process. If no process "
                    "works in it, release it: mv %s %s",
                    *(folder, folder, attempt_dir / "free__released"),
                )
                waited = True
            time.sleep(_WAIT_S)
            continue
        if len(names) > 1 or why_held(names[0]) is not None:
            raise RuntimeError(refusal(attempt_dir, names))
        mine = attempt_dir / f"{owner()}__{_tok()}"
        # A lost race and a retried rename both fail; the listing says which.
        with suppress(FileNotFoundError):
            os.rename(attempt_dir / names[0], mine)
        with suppress(FileNotFoundError):
            if mine.name in entries(attempt_dir):
                return mine


def release(attempt: Path) -> None:
    """Hand a claimed working folder back: free if it has work in it, else gone."""
    with suppress(FileNotFoundError):  # published, or deleted
        if os.listdir(attempt):
            os.rename(attempt, attempt.with_name(f"free__{_tok()}"))
            return
        os.rmdir(attempt)
    with suppress(OSError):  # someone seeded it again
        os.rmdir(attempt.parent)


@contextmanager
def claimed(attempt_dir: Path) -> Generator[Path]:
    """Hold the working folder of attempts/<run>/ for the body, then release it."""
    attempt = claim(attempt_dir)
    try:
        yield attempt
    finally:
        release(attempt)


def free_after_death(attempt_dir: Path, prefix: str) -> None:
    """Release the folder a dead child of this process held, named with ``prefix``.

    Its parent saw it die, which is all the liveness a folder needs.
    """
    with suppress(FileNotFoundError):
        for name in entries(attempt_dir):
            if name.startswith(prefix + "__"):
                with suppress(FileNotFoundError):
                    os.rename(attempt_dir / name, attempt_dir / f"free__{_tok()}")


def trash(folder: Path, attempt: Path) -> Path:
    """Move ``folder`` beside ``attempt`` under a dotted name that listings skip.

    One rename, so no reader finds it half deleted; delete what it returns.
    """
    moved = attempt.with_name(f".trash__{_tok()}")
    os.rename(folder, moved)
    return moved
