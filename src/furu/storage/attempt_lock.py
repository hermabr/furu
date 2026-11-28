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
gone, or that the run of it which took the folder, the restart count in its
name, was followed by another: a requeued job runs again. A process's
outside Slurm is freed by the worker that started it and saw it die, and
otherwise only by hand.
"""

from __future__ import annotations

import errno
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
_RACE = frozenset({errno.EEXIST, errno.ENOTEMPTY, errno.ENOENT, errno.ESTALE})


class HeldError(RuntimeError):
    """A run's working folder is another's, or may be."""


def _me() -> str:
    """This process as an owner: the Slurm job it runs in, if it runs in one."""
    if job := os.environ.get("SLURM_JOB_ID"):
        restart = os.environ.get("SLURM_RESTART_COUNT") or "0"
        if not job.isdigit() or not restart.isdigit():
            raise ValueError(
                f"invalid SLURM_JOB_ID={job!r} or SLURM_RESTART_COUNT={restart!r}"
            )
        cluster = slurm.cluster()
        if not cluster:
            raise RuntimeError(
                f"SLURM_JOB_ID={job} is set, but the Slurm cluster name could not "
                "be determined from SLURM_CLUSTER_NAME or `scontrol show config`"
            )
        return f"slurm__{cluster}__{job}__{restart}"
    return f"local__{socket.gethostname()}__{os.getpid()}"


def current(run: Path) -> Path | None:
    """The run's working folder under the name it has now, if it has one."""
    names = sorted(list_names(run))
    return run / names[0] if names else None


def _only(run: Path) -> str | None:
    """The name of the run's working folder, if it has one."""
    for _ in range(3):  # a listing taken during a rename can show both names
        names = sorted(list_names(run))
        if len(names) <= 1:
            return names[0] if names else None
    raise HeldError(
        f"{run} holds {len(names)} folders, {', '.join(names)}, where a run has "
        "one working folder. Remove the ones nothing is writing to."
    )


def free(run: Path) -> Path | None:
    """The run's working folder if it may be taken, or None if it has none.

    Raises HeldError if the folder is another's, or may be.
    """
    if (name := _only(run)) is None:
        return None
    match name.split("__"):
        case ["free", tok] if tok:
            return run / name
        case ["slurm", cluster, job, restart, tok] if (
            cluster and job.isdigit() and restart.isdigit() and tok
        ):
            if slurm.gone(cluster, job, int(restart)):
                return run / name
            whose = (
                f"is the working folder of Slurm job {job} on cluster {cluster} "
                f"(run {restart}), which Slurm does not report from here as "
                "ended with all its processes gone, or as requeued"
            )
        case ["local", host, pid, tok] if host and pid.isdigit() and tok:
            whose = (
                f"is the working folder of process {pid} on {host}, and furu "
                "does not ask whether a process outside Slurm is alive"
            )
        case _:
            whose = "has a name that does not say whose working folder it is"
    raise HeldError(
        f"{run / name} {whose}. If nothing is writing to it, free it by hand: "
        f"mv {run / name} {run}/free__manual"
    )


def trash(folder: Path, *, into: Path | None = None) -> OSError | None:
    """Delete ``folder``: one rename takes it out of sight whole, whatever
    becomes of deleting it."""
    dest = (into or folder.parent.parent) / f".trash__{secrets.token_hex(8)}"
    err: OSError | None = None
    try:
        os.rename(folder, dest)
    except OSError as exc:
        err = exc
    shutil.rmtree(dest, ignore_errors=True)
    return err


def acquire(run: Path, *, token: str | None = None) -> tuple[Path, os.stat_result]:
    """Take the run's working folder for this process, making one if it has none.

    Returns the folder and its stat, which ``publish`` asks for. Raises
    HeldError if the folder is another's, or may be. ``token`` ends the
    folder's name: a worker gives its child one, to find the folder by if the
    child dies.
    """
    mine = run / f"{_me()}__{token or secrets.token_hex(8)}"
    err: OSError | None = None
    if (existing := free(run)) is not None:
        try:
            os.rename(existing, mine)
        except OSError as exc:
            err = exc
    else:
        # Made whole beside the run's folder and renamed onto it, which works
        # only while that is missing or empty: of the workers that start a run
        # at once, one's folder is the one in it.
        run.parent.mkdir(parents=True, exist_ok=True)
        seed = run.parent / f".seed__{secrets.token_hex(8)}"
        seed.mkdir()
        (seed / mine.name).mkdir()
        try:
            os.rename(seed, run)
        except OSError as exc:
            err = exc
        shutil.rmtree(seed, ignore_errors=True)
    if (name := _only(run)) != mine.name:
        if err is not None and err.errno not in _RACE:
            raise err
        now = f"is now {run / name}" if name else "is gone"
        raise HeldError(
            f"did not get {run}: another worker took it at the same moment. "
            f"Its working folder {now}"
        ) from err
    try:
        return mine, mine.stat()
    except BaseException:
        release(mine)
        raise


def release(mine: Path) -> None:
    """Give up a working folder this process took.

    An empty folder is removed; one with files stays under a ``free__`` name
    so that any worker may take it and go on from what is in it.
    """
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
        current_stat = mine.stat()
    except FileNotFoundError:
        current_stat = None
    if current_stat is None or not os.path.samestat(current_stat, claimed):
        if current_stat is not None:
            trash(mine)
        raise HeldError(
            f"{mine} is no longer the folder this process took: it was renamed "
            "while its create() ran, so nothing is published from here"
        )
    # Scratch is not part of a run. It is moved out whole, which a crash
    # cannot leave half done as deleting it in place would.
    dest.parent.mkdir(parents=True, exist_ok=True)
    moved_scratch = mine.parent.parent / f".trash__{secrets.token_hex(8)}"
    with suppress(FileNotFoundError):
        os.rename(scratch_dir_in(mine), moved_scratch)
    try:
        os.rename(mine, dest)
    except OSError:
        if dest.name not in list_names(dest.parent):
            with suppress(OSError):
                os.rename(moved_scratch, scratch_dir_in(mine))
            raise
        # The run is there: this rename happened and its answer was lost, or
        # another worker published the run first and this folder is left over.
        if mine.name in list_names(mine.parent):
            trash(mine)
    shutil.rmtree(moved_scratch, ignore_errors=True)
    with suppress(OSError):
        mine.parent.rmdir()


def discard(run: Path) -> None:
    """Delete the run's working folder, if it has one.

    Raises HeldError if the folder is another's, or is a worker's by the time
    this is done.
    """
    err = trash(existing) if (existing := free(run)) is not None else None
    if names := list_names(run):
        if err is not None and err.errno not in _RACE:
            raise err
        raise HeldError(
            f"the working folder of {run} was not deleted: a worker may have "
            f"taken the run, which now holds {names[0]}"
        ) from err
    with suppress(OSError):
        run.rmdir()
