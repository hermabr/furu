"""A run's working folder is named for its owner, and that name is the lock.

``attempts/<run>/`` holds the run's one working folder and nothing else:

    free__<tok>                                  nobody's: any worker may take it
    slurm__<cluster>__<jobid>__<restart>__<tok>  a Slurm job's
    local__<host>__<pid>__<tok>                  a process's outside Slurm

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
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from furu import slurm
from furu.logging import get_logger
from furu.storage._layout import list_names, scratch_dir_in

logger = get_logger()


class HeldError(RuntimeError):
    """A run's working folder is another's, or may be."""


def new_token() -> str:
    return secrets.token_hex(8)


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
        return f"slurm__{cluster}__{job}__{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    return f"local__{socket.gethostname()}__{os.getpid()}"


def entries(run: Path) -> list[str]:
    """What is in a run's ``attempts/<run>/`` now: its working folder, or nothing."""
    return sorted(list_names(run))


def current(run: Path) -> Path | None:
    """The run's working folder under the name it has now, if it has one."""
    names = entries(run)
    return run / names[0] if names else None


def _only(run: Path) -> str | None:
    """The name of the run's working folder, if it has one."""
    for _ in range(3):  # a listing taken during a rename can show both names
        names = entries(run)
        if len(names) <= 1:
            return names[0] if names else None
    raise HeldError(
        f"{run} holds {len(names)} folders, {', '.join(names)}, where a run has "
        "one working folder. Remove the ones nothing is writing to."
    )


def _why_held(run: Path, name: str) -> str | None:
    """Why the working folder called ``name`` may not be taken; None if it may."""
    match name.split("__"):
        case ["free", tok] if tok:
            return None
        case ["slurm", cluster, job, restart, tok] if (
            cluster and job.isdigit() and restart.isdigit() and tok
        ):
            if slurm.gone(cluster, job):
                return None
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
    return (
        f"{run / name} {whose}. If nothing is writing to it, free it by hand: "
        f"mv {run / name} {run}/free__manual"
    )


def held(run: Path) -> str | None:
    """Why the run cannot be started or continued now; None if it can."""
    try:
        name = _only(run)
    except HeldError as exc:
        return str(exc)
    return _why_held(run, name) if name else None


def _take(run: Path, dest: Path) -> bool:
    """Rename the run's working folder to ``dest``; False if it has none.

    Raises HeldError if the folder is another's. The rename can lose to
    another worker's and can land with its answer lost, so the caller lists
    again to see what happened.
    """
    name = _only(run)
    if name is None:
        return False
    if why := _why_held(run, name):
        raise HeldError(why)
    with suppress(OSError):
        os.rename(run / name, dest)
    return True


def _remove(folder: Path) -> None:
    """Delete a folder: one rename takes it out of sight whole, whatever
    becomes of deleting it."""
    trash = folder.parent.parent / f".trash__{new_token()}"
    with suppress(OSError):
        os.rename(folder, trash)
    shutil.rmtree(trash, ignore_errors=True)


def acquire(
    run: Path, published: Callable[[], bool], *, token: str | None = None
) -> tuple[Path, os.stat_result] | None:
    """Take the run's working folder for this process, making one if it has none.

    Returns the folder and its stat, which ``publish`` asks for; None when
    ``published`` says, once the folder is taken, that the run is stored after
    all. Raises HeldError if the folder is another's, or may be. ``token`` ends
    the folder's name: a worker gives its child one, to find the folder by if
    the child dies.
    """
    mine = run / f"{me()}__{token or new_token()}"
    if not _take(run, mine):
        # Made whole beside the run's folder and renamed onto it, which works
        # only while that is missing or empty: of the workers that start a run
        # at once, one's folder is the one in it.
        run.parent.mkdir(parents=True, exist_ok=True)
        seed = run.parent / f".seed__{new_token()}"
        seed.mkdir()
        (seed / mine.name).mkdir()
        with suppress(OSError):
            os.rename(seed, run)
        shutil.rmtree(seed, ignore_errors=True)
    if (name := _only(run)) != mine.name:
        now = f"is now {run / name}" if name else "is gone"
        raise HeldError(
            f"did not get {run}: another worker took it at the same moment, or "
            f"the rename failed. Its working folder {now}"
        )
    try:
        claimed = mine.stat()
        # Whoever held the folder before may have published the run since this
        # process last looked, and a run is published once.
        already = published()
    except BaseException:
        _undo(mine, run)
        raise
    if already:
        _undo(mine, run)
        return None
    return mine, claimed


def _undo(mine: Path, run: Path) -> None:
    try:
        mine.rmdir()  # made here
    except OSError:
        release(mine)  # taken with what earlier attempts left in it
    with suppress(OSError):
        run.rmdir()


def release(mine: Path) -> None:
    """Give up a working folder this process took, so that any worker may take
    it and go on from what is in it."""
    try:
        os.rename(mine, mine.with_name(f"free__{new_token()}"))
    except OSError as exc:
        if mine.name in entries(mine.parent):
            logger.warning("could not release %s, which stays held: %s", mine, exc)


def release_child(run: Path, token: str) -> None:
    """Give up the working folder a child died holding, found by the token it
    was given. Only for a child this process killed and waited for: nothing
    else here says that a process is dead."""
    for name in entries(run):
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
    if not same:
        _remove(mine)
        raise HeldError(
            f"{mine} is no longer the folder this process took: it was renamed "
            "while its create() ran, so nothing is published from here"
        )
    # Scratch is not part of a run. It is moved out whole, which a crash
    # cannot leave half done as deleting it in place would.
    dest.parent.mkdir(parents=True, exist_ok=True)
    trash = mine.parent.parent / f".trash__{new_token()}"
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
        if mine.name in entries(mine.parent):
            _remove(mine)
    shutil.rmtree(trash, ignore_errors=True)
    with suppress(OSError):
        mine.parent.rmdir()


def discard(run: Path) -> None:
    """Delete the run's working folder, if it has one.

    Raises HeldError if the folder is another's, or is a worker's by the time
    this is done.
    """
    trash = run.parent / f".trash__{new_token()}"
    if _take(run, trash):
        shutil.rmtree(trash, ignore_errors=True)
    if names := entries(run):
        raise HeldError(
            f"the working folder of {run} was not deleted: a worker may have "
            f"taken the run, which now holds {names[0]}"
        )
    with suppress(OSError):
        run.rmdir()
