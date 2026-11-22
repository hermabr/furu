"""attempts/<run>/: a run's working folder, whose name says who may work in it.

attempts/<run>/ holds at most one folder whose name has no leading dot, the
run's working folder. Its name is the lock:

    free__<tok>                                    nobody's: any worker may take it
    slurm__<cluster>__<job id>__<restart>__<tok>   a Slurm job's
    local__<host>__<pid>__<tok>                    a process's outside Slurm

A worker takes the folder by renaming it to its own name, and gives it back
by renaming it to free__<tok>. Of two renames of one name one wins, on NFS
too, and <tok> is new on every rename, so a worker never renames a later
folder that happens to have the name it read. Publishing renames the folder
into runs/. Dotted names are folders being thrown away.

A folder is taken from its holder only when the holder surely runs no more:
a Slurm job that squeue no longer lists and that accounting says ended
cleanly. A process outside Slurm is never presumed dead; a person moves its
folder to free__manual, or removes it.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import subprocess
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Literal

from furu import slurm
from furu.storage._layout import scratch_dir_in

type Owner = Literal["adoptable", "held", "unknown"]

SEED = "free__seed"

# How a Slurm job may have ended for its folders to be taken: it can run no
# more code. NODE_FAIL and BOOT_FAIL are not here, nor is a job whose kill
# failed: their processes may still be running on the node.
_CLEAN_ENDS = frozenset(
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


class HeldError(RuntimeError):
    """The working folder of a run is held, or may be, by another worker."""


def token() -> str:
    return secrets.token_hex(8)


def names(folder: Path) -> list[str]:
    """What a listing of ``folder`` shows now; nothing when there is no folder.

    A listing asks the server, where a lookup by name may be answered from
    what this machine remembers for up to a minute.
    """
    try:
        return os.listdir(folder)
    except FileNotFoundError:
        return []


def listed(path: Path) -> bool:
    return path.name in names(path.parent)


def entries(run: Path) -> list[str]:
    """The working folders in attempts/<run>/: one, unless a race left more."""
    # A folder made by a race (SEED) is empty; the others may hold checkpoints.
    return sorted(
        (name for name in names(run) if not name.startswith(".")),
        key=lambda name: (name == SEED, name),
    )


def current(run: Path) -> Path:
    """The working folder in attempts/<run>/ now, or attempts/<run>/ when it has none."""
    found = entries(run)
    return run / found[0] if found else run


def me() -> str:
    """What this process names the folders it takes, but for the token."""
    if job := os.environ.get("SLURM_JOB_ID"):
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        return f"slurm__{slurm.cluster() or ''}__{job}__{restart}"
    return f"local__{socket.gethostname()}__{os.getpid()}"


def owner(name: str) -> Owner:
    """Whether this process may take the working folder ``name``."""
    parts = name.split("__")
    if not all(parts):
        return "unknown"
    match parts:
        case ["free", _]:
            return "adoptable"
        case ["local", _, _, _]:
            return "held"
        case ["slurm", cluster, job, _, _] if cluster == slurm.cluster():
            return _slurm_owner(job)
    return "unknown"


def _slurm_owner(job: str) -> Owner:
    """Adoptable only when squeue no longer lists the job and it ended cleanly."""
    try:
        queued = slurm.run(
            "squeue", "--all", "--noheader", "-O", "JobIDRaw:32,State:32", "-j", job
        )
    except subprocess.CalledProcessError as exc:
        # Older Slurm says so of a job it has purged; newer lists nothing.
        if "Invalid job id" not in (exc.stderr or ""):
            return "unknown"
        queued = ""
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if queued.strip():
        return "held"
    try:
        accounted = slurm.run(
            "sacct", "-D", "-X", "--noheader", "--parsable2",
            "-o", "JobIDRaw,State,Reason", "-j", job,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    rows = [line.split("|") for line in accounted.splitlines() if line.strip()]
    clean = rows and all(
        len(row) == 3
        and row[0] == job
        # "CANCELLED by <uid>" is CANCELLED.
        and row[1].partition(" by ")[0] in _CLEAN_ENDS
        and row[2] != "Kill task failed"
        for row in rows
    )
    return "adoptable" if clean else "unknown"


def holder(run: Path, found: Iterable[str] | None = None) -> str | None:
    """Why this process may not take the working folder of attempts/<run>/, if so."""
    for name in entries(run) if found is None else found:
        match owner(name):
            case "adoptable":
                continue
            case "held" if name.startswith("local__"):
                return (
                    f"{run / name} is held by process {name.split('__')[2]} on "
                    f"{name.split('__')[1]}. If that process is dead, move the "
                    f"folder to {run / 'free__manual'} to go on from what it "
                    "left, or remove it to start over"
                )
            case "held":
                return f"{run / name} is held by a Slurm job that is still queued"
            case _:
                return (
                    f"{run / name} may be held: its owner cannot be shown to have "
                    "stopped. If it has, move the folder to "
                    f"{run / 'free__manual'}, or remove it"
                )
    return None


def acquire(run: Path, published: Iterable[Path] = ()) -> Path | None:
    """Take the working folder in attempts/<run>/ for this process, making one if none.

    None when one of ``published``, the folders the run is published as, is
    there. Raises HeldError when another worker holds the folder, or may.
    """
    published = list(published)
    if any(map(listed, published)):
        return None
    for _ in range(3):
        found = entries(run)
        if not found:
            # Publishing empties attempts/<run>/ and removes it, so look again.
            if any(map(listed, published)):
                return None
            with suppress(FileExistsError, FileNotFoundError):
                (run / SEED).mkdir(parents=True)
            if not (found := entries(run)):
                continue
        if why := holder(run, found):
            raise HeldError(why)
        mine = run / f"{me()}__{token()}"
        try:
            (run / found[0]).rename(mine)
        except OSError:
            pass  # taken first, or NFS retried a rename that went through
        if listed(mine):
            if SEED in found[1:]:  # made by a race, and empty
                with suppress(OSError):
                    (run / SEED).rmdir()
            return mine
    raise HeldError(f"could not take the working folder in {run}: others took it")


def release(attempt: Path) -> None:
    """Give the working folder back to any worker, unless it is published."""
    with suppress(FileNotFoundError):
        attempt.rename(attempt.with_name(f"free__{token()}"))


def publish(attempt: Path, run: Path) -> None:
    """Rename the working folder to ``run`` in runs/, and throw away what is left.

    A run that is there is never replaced: this attempt is thrown away instead.
    A rename that may have failed leaves the folder, the one finished copy.
    """
    trash = attempt.with_name(f".trash__{token()}")
    with suppress(FileNotFoundError):
        scratch_dir_in(attempt).rename(trash)
    run.parent.mkdir(parents=True, exist_ok=True)
    try:
        attempt.rename(run)
    except OSError:
        if not listed(run):
            raise
        if listed(attempt):  # another worker published the run first
            lost = attempt.with_name(f".trash__{token()}")
            attempt.rename(lost)
            shutil.rmtree(lost, ignore_errors=True)
    shutil.rmtree(trash, ignore_errors=True)
    with suppress(OSError):
        attempt.parent.rmdir()
