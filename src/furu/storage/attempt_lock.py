"""A run's working folder is named for its owner, and that name is the lock.

``attempts/<run>/`` holds at most one folder whose name does not start with a
dot, the run's working folder:

    free__<tok>                                  nobody's: any worker may take it
    slurm__<cluster>__<jobid>__<restart>__<tok>  a Slurm job's
    local__<host>__<pid>__<tok>                  a process's outside Slurm

Taking, releasing and publishing the folder are each one rename, which has one
winner however many machines try. ``<tok>`` is new on every rename, so no name
is used twice and a worker that read an older name cannot take a newer folder
by it. A name that starts with a dot is a folder on its way to being deleted.

Nothing in the folder says whether its owner is alive, and no clock is read. A
Slurm job's folder may be taken once Slurm says the job ended cleanly. A
process's outside Slurm is freed by the worker that started it and saw it die,
and otherwise only by hand.
"""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
from enum import Enum
from pathlib import Path

from furu.logging import get_logger
from furu.worker.backends.slurm import command

logger = get_logger()

_SEED = "free__seed"
# A try is lost only to another worker that takes or removes the folder at the
# same moment, and that worker's folder then refuses the next try.
_TRIES = 5
# The states of a job whose processes Slurm saw end. Not NODE_FAIL or
# BOOT_FAIL: a node that stopped answering can still be running them.
_ENDED_CLEANLY = frozenset(
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


class AttemptHeld(RuntimeError):
    """A run's working folder could not be taken: it is, or may be, another's."""


class Owner(Enum):
    """Whether a working folder may be taken, going by its name."""

    ADOPTABLE = "adoptable"  # nobody's, or its owner is known to be gone
    HELD = "held"  # its owner is alive, or cannot be asked about
    UNKNOWN = "unknown"  # nothing says whether its owner is alive


def new_token() -> str:
    return secrets.token_hex(8)


def _me(pid: int | None = None) -> str:
    """This process as an owner, or its child ``pid``: the Slurm job it runs
    in, if it runs in one."""
    job, cluster = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_CLUSTER_NAME")
    if job and cluster:
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        return f"slurm__{cluster}__{job}__{restart}"
    return f"local__{socket.gethostname()}__{pid or os.getpid()}"


def classify(name: str) -> Owner:
    """Whether the working folder called ``name`` may be taken.

    A name of any other shape than the three is never taken.
    """
    match name.split("__"):
        case ["free", tok] if tok:
            return Owner.ADOPTABLE
        case ["local", host, pid, tok] if host and pid.isdigit() and tok:
            return Owner.HELD
        case ["slurm", cluster, job, restart, tok] if (
            cluster and job.isdigit() and restart.isdigit() and tok
        ):
            return _slurm_job(cluster, job)
        case _:
            return Owner.UNKNOWN


def _slurm_job(cluster: str, job: str) -> Owner:
    """Whether Slurm says that ``job`` ended and left no process behind.

    The job has to be gone from the queue and every row accounting has of it
    has to be a clean end. Anything else, a command that fails or an answer
    that is not understood, leaves the folder with the job.
    """
    if cluster != command.cluster():
        return Owner.UNKNOWN
    try:
        queued = command.slurm(
            "squeue", "--all", "--noheader", "-O", "JobID:32,State:32", "-j", job
        )
    except subprocess.CalledProcessError as exc:
        # How the queue answers for a job it no longer remembers.
        if "Invalid job id" not in (exc.stderr or ""):
            return Owner.UNKNOWN
    except (OSError, subprocess.SubprocessError):
        return Owner.UNKNOWN
    else:
        if queued.strip():  # pending, running, completing, ...
            return Owner.HELD
    try:
        # -D: a requeued job has a row for each time it ran.
        accounted = command.slurm(
            *("sacct", "-D", "-X", "--noheader", "--parsable2"),
            *("-o", "JobIDRaw,State,Reason", "-j", job),
        )
    except (OSError, subprocess.SubprocessError):
        return Owner.UNKNOWN
    rows = [line.split("|") for line in accounted.splitlines() if line.strip()]
    clean = all(
        len(row) == 3
        # "CANCELLED by <uid>" is CANCELLED.
        and row[1].split(" by ")[0] in _ENDED_CLEANLY
        # The job ended in accounting while a process of it could not be killed.
        and row[2] != "Kill task failed"
        for row in rows
    )
    # No row of the job yet: accounting is behind the queue.
    if clean and any(row[0] == job for row in rows):
        return Owner.ADOPTABLE
    return Owner.UNKNOWN


def why_not_free(folder: Path) -> str | None:
    """Why the working folder ``folder`` may not be taken; None if it may."""
    owner = classify(folder.name)
    if owner is Owner.ADOPTABLE:
        return None
    by_hand = (
        f"`mv {folder} {folder.parent / 'free__manual'}` to go on from what it "
        f"wrote, or `rm -r {folder}` to start over"
    )
    match folder.name.split("__"):
        case ["local", host, pid, _]:
            return (
                f"{folder} is the working folder of process {pid} on {host}, and "
                "furu cannot tell whether a process outside Slurm is alive. If "
                f"it is dead, {by_hand}."
            )
        case ["slurm", _, job, _, _] if owner is Owner.HELD:
            return (
                f"{folder} is the working folder of Slurm job {job}, which is "
                "still queued or running."
            )
        case ["slurm", cluster, job, _, _]:
            return (
                f"{folder} is the working folder of Slurm job {job} on cluster "
                f"{cluster}, and Slurm does not say from here that the job ended "
                "with all its processes gone. If nothing of it is running, "
                f"{by_hand}."
            )
        case _:
            return (
                f"{folder} is in the place of a working folder, and its name does "
                f"not say whose it is. If nothing is writing to it, {by_hand}."
            )


def entries(parent: Path) -> list[str]:
    """The working folders in a run's ``attempts/<run>/``, as a listing shows now.

    At most one, unless one was put there by hand. A listing asks the file
    server, where a lookup by name can be answered from what this machine
    remembers.
    """
    try:
        names = os.listdir(parent)
    except FileNotFoundError:
        return []
    return sorted(name for name in names if not name.startswith("."))


def current(parent: Path) -> Path | None:
    """The run's working folder under the name it has now, if it has one."""
    names = entries(parent)
    return parent / names[0] if names else None


def blocked(parent: Path, names: list[str] | None = None) -> str | None:
    """Why the run cannot be started or continued now; None if it can.

    ``names`` are its working folders, when the caller has listed them.
    """
    return next(
        (
            why
            for name in (entries(parent) if names is None else names)
            if (why := why_not_free(parent / name)) is not None
        ),
        None,
    )


def acquire(parent: Path) -> Path:
    """Take the run's working folder for this process, making one if there is none.

    Raises AttemptHeld if the folder is another's, or may be.
    """
    mine = f"{_me()}__{new_token()}"
    error: OSError | None = None
    for _ in range(_TRIES):
        parent.mkdir(parents=True, exist_ok=True)
        names = entries(parent)
        if not names:
            # One mkdir wins; a loser finds the winner's folder, under the name
            # it has by then. The parent is gone if its run was just published.
            try:
                (parent / _SEED).mkdir()
            except (FileExistsError, FileNotFoundError):
                pass
            names = entries(parent)
        # The rename below is believed only once a listing shows it: its error
        # can be the lost answer to a rename that happened.
        if mine in names:
            return parent / mine
        if (why := blocked(parent, names)) is not None:
            raise AttemptHeld(why)
        if names:
            try:
                os.rename(parent / names[0], parent / mine)
            except OSError as exc:
                error = exc
    raise AttemptHeld(
        f"could not take a working folder in {parent} in {_TRIES} tries"
    ) from error


def release_child(parent: Path, pid: int) -> None:
    """Give up the working folder in ``parent`` that child ``pid`` died holding.

    Only for a child this process has waited for: nothing else here says that
    a process is dead.
    """
    for name in entries(parent):
        if name.startswith(f"{_me(pid)}__"):
            release(parent / name)


def release(attempt: Path) -> None:
    """Give up a working folder this process took, so that any worker may take
    it and go on from what is in it."""
    try:
        os.rename(attempt, attempt.with_name(f"free__{new_token()}"))
    except FileNotFoundError:
        pass  # taken from this process by hand
    except OSError as exc:
        logger.warning("could not release %s, which stays held: %s", attempt, exc)
