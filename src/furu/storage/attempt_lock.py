"""A run's working folder is named for its owner, and that name is the lock.

``attempts/<run>/`` holds the run's one working folder and nothing else:

    free__<tok>                       nobody's: any worker may take it
    slurm__<cluster>__<jobid>__<tok>  a Slurm job's
    local__<host>__<pid>__<tok>       a process's outside Slurm

Taking, releasing and publishing the folder are each one rename, which has one
winner however many machines try. A rename is believed only once a listing
shows what it did: its error can be the lost answer to a rename that happened.
``<tok>`` is new on every rename, so no name is used twice. A folder on its
way in or out has a dotted name in ``attempts/``, never in ``attempts/<run>/``.

Nothing in the folder says whether its owner is alive, no clock is read and
nobody waits. A Slurm job's folder may be taken once Slurm says the job is
gone. A process's outside Slurm is freed by the worker that started it and saw
it die, and otherwise only by hand.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
from contextlib import suppress
from pathlib import Path

from furu import slurm
from furu.logging import get_logger
from furu.storage._layout import list_names, scratch_dir_in

logger = get_logger()


class HeldError(RuntimeError):
    """A run's working folder is another's, or may be."""


def me() -> str:
    """This process as an owner: the Slurm job it runs in, if it runs in one."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if job.isdigit():
        cluster = slurm.cluster()
        if not cluster:
            raise RuntimeError(
                f"SLURM_JOB_ID={job} is set, but the Slurm cluster name could not "
                "be determined from SLURM_CLUSTER_NAME or `scontrol show config`"
            )
        return f"slurm__{cluster}__{job}"
    return f"local__{socket.gethostname()}__{os.getpid()}"


def current(run: Path) -> Path | None:
    """The run's working folder under the name it has now, if it has one."""
    names = sorted(list_names(run))
    return run / names[0] if names else None


def takeable(run: Path) -> Path | None:
    """The run's working folder, which may be taken; None if the run has none.

    Raises HeldError if the folder is another's, or may be.
    """
    for _ in range(3):  # a listing taken during a rename can show both names
        names = sorted(list_names(run))
        if not names:
            return None
        if len(names) == 1:
            break
    else:
        raise HeldError(
            f"{run} holds {len(names)} folders, {', '.join(names)}, where a run has "
            "one working folder. Remove the ones nothing is writing to."
        )
    folder = run / names[0]
    match folder.name.split("__"):
        case ["free", tok] if tok:
            return folder
        case ["slurm", cluster, job, tok] if cluster and job.isdigit() and tok:
            if slurm.gone(cluster, job):
                return folder
            whose = (
                f"is the working folder of Slurm job {job} on cluster {cluster}, "
                "which Slurm does not report from here as ended with all its "
                "processes gone"
            )
        case ["local", host, pid, tok] if host and pid.isdigit() and tok:
            whose = (
                f"is the working folder of process {pid} on {host}, and furu "
                "does not ask whether a process outside Slurm is alive"
            )
        case _:
            whose = "has a name that does not say whose working folder it is"
    raise HeldError(
        f"{folder} {whose}. If nothing is writing to it, free it by hand: "
        f"mv {folder} {run}/free__manual"
    )


def remove(folder: Path, attempts: Path) -> None:
    """Delete a folder, if it is there: one rename into ``attempts`` takes it
    out of sight whole, whatever becomes of deleting it."""
    trash = attempts / f".trash__{secrets.token_hex(8)}"
    with suppress(FileNotFoundError):
        os.rename(folder, trash)
    shutil.rmtree(trash, ignore_errors=True)


def acquire(run: Path, token: str | None = None) -> tuple[Path, os.stat_result]:
    """Take the run's working folder for this process, making one if it has none.

    Returns the folder and its stat, which ``publish`` asks for. Raises
    HeldError if the folder is another's, or may be. ``token`` ends the
    folder's name: a worker gives its child one, to find the folder by if the
    child dies.
    """
    mine = run / f"{me()}__{token or secrets.token_hex(8)}"
    if folder := takeable(run):
        with suppress(OSError):
            os.rename(folder, mine)
    else:
        # Made whole beside the run's folder and renamed onto it, which works
        # only while that is missing or empty: of the workers that start a run
        # at once, one's folder is the one in it.
        run.parent.mkdir(parents=True, exist_ok=True)
        seed = run.parent / f".seed__{secrets.token_hex(8)}"
        seed.mkdir()
        (seed / mine.name).mkdir()
        with suppress(OSError):
            os.rename(seed, run)
        shutil.rmtree(seed, ignore_errors=True)
    if (now := current(run)) != mine:
        raise HeldError(
            f"did not get {run}: another worker took it at the same moment, or "
            f"the rename failed. Its working folder "
            f"{f'is now {now}' if now else 'is gone'}"
        )
    return mine, mine.stat()


def release(mine: Path) -> None:
    """Give up a working folder this process took, so that any worker may take
    it and go on from what is in it. An empty one is deleted."""
    try:
        mine.rmdir()
    except OSError:
        try:
            os.rename(mine, mine.with_name(f"free__{secrets.token_hex(8)}"))
        except OSError as exc:
            if mine.name in list_names(mine.parent):
                logger.warning("could not release %s, which stays held: %s", mine, exc)
    else:
        with suppress(OSError):
            mine.parent.rmdir()


def release_child(run: Path, token: str) -> None:
    """Give up the working folder a child died holding, found by the token it
    was given. Only for a child this process killed and waited for: nothing
    else here says that a process is dead."""
    for name in list_names(run):
        if name.endswith(f"__{token}") and not name.startswith("free__"):
            release(run / name)


def publish(mine: Path, dest: Path, claimed: os.stat_result) -> None:
    """Rename a finished working folder to ``dest``, the run in runs/.

    One rename, so readers see the whole run or none of it, and a run that is
    already there is never replaced. A rename that fails leaves the working
    folder as it was, the only copy of what was computed, unless a listing
    shows the run in runs/ after all.
    """
    # A folder that was taken from this process is gone under this name,
    # unless its create() made it again by writing to a path with its parents.
    try:
        same = os.path.samestat(mine.stat(), claimed)
    except FileNotFoundError:
        same = False
    attempts = mine.parent.parent
    if not same:
        remove(mine, attempts)
        raise HeldError(
            f"{mine} is no longer the folder this process took: it was renamed "
            "while its create() ran, so nothing is published from here"
        )
    # Scratch is not part of a run. It is moved out whole, which a crash
    # cannot leave half done as deleting it in place would.
    dest.parent.mkdir(parents=True, exist_ok=True)
    trash = attempts / f".trash__{secrets.token_hex(8)}"
    with suppress(FileNotFoundError):
        os.rename(scratch_dir_in(mine), trash)
    try:
        os.rename(mine, dest)
    except OSError:
        if dest.name not in list_names(dest.parent):
            with suppress(OSError):
                os.rename(trash, scratch_dir_in(mine))
            raise
        # The run is there: this rename happened and its answer was lost, or
        # another worker published the run first and this folder is left over.
        remove(mine, attempts)
    shutil.rmtree(trash, ignore_errors=True)
    with suppress(OSError):
        mine.parent.rmdir()


def discard(run: Path) -> None:
    """Delete the run's working folder, if it has one.

    Raises HeldError if the folder is another's, or is a worker's by the time
    this is done.
    """
    if folder := takeable(run):
        remove(folder, run.parent)
    if now := current(run):
        raise HeldError(
            f"the working folder of {run} was not deleted: a worker may have "
            f"taken the run, which now holds {now.name}"
        )
    with suppress(OSError):
        run.rmdir()
