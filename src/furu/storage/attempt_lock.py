"""attempts/<run>/: a run's working folder, whose name says who may work in it.

attempts/<run>/ holds one folder, the run's working folder, or none. Its
name is the lock:

    free__<tok>                                   nobody's: any worker may take it
    slurm__<cluster>__<job>__<restart>__<tok>     a Slurm job's
    local__<host>__<pid>__<tok>                   a process's outside Slurm

A worker takes the folder by renaming it to its own name, and gives it back by
renaming it to free__<tok>. Of two renames of one name one wins, on NFS too,
and <tok> is new on every rename, so a worker never renames a later folder
that happens to have a name it read. A run with no folder gets one with the
folder that holds it: attempts/.seed__<tok>/<name>/ is renamed to
attempts/<run>/, which only works while attempts/<run>/ is empty or missing.
Publishing renames the working folder into runs/. Folders being thrown away
are dotted ones in attempts/ itself.

A folder is taken from its holder only when the holder surely runs no more: a
Slurm job that squeue no longer lists and that accounting says ended cleanly.
A process outside Slurm is never presumed dead; a person renames its folder
to free__manual, or removes it. Nothing here waits: a held folder is an error
that says what holds it.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from furu import slurm
from furu.logging import get_logger
from furu.storage._layout import list_names, log_path_in, scratch_dir_in

logger = get_logger()


class HeldError(RuntimeError):
    """The working folder of a run is held by another worker, or may be."""


@dataclass(frozen=True)
class Claim:
    """A working folder this process took, and the folder it took."""

    path: Path
    # Which folder: one made at the same path after this one was taken away
    # is another, and is never published.
    stat: os.stat_result


def new_token() -> str:
    return secrets.token_hex(8)


def me() -> str:
    """What this process names the folders it takes, but for the token."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if job.isdigit() and (cluster := slurm.cluster()):
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        return f"slurm__{cluster}__{job}__{restart}"
    return f"local__{socket.gethostname()}__{os.getpid()}"


def entries(run: Path) -> list[str]:
    """The names in attempts/<run>/: one, or none, unless a rename is under way."""
    return sorted(list_names(run))


def current(run: Path) -> Path | None:
    """The working folder in attempts/<run>/ now, if it has one."""
    names = entries(run)
    return run / names[0] if names else None


def log_in(run: Path) -> Path:
    """The log of the working folder in attempts/<run>/; the folder itself if none."""
    found = current(run)
    return run if found is None else log_path_in(found)


def _why_held(run: Path, name: str) -> str | None:
    """Why this process may not take the working folder ``name``, if it may not."""
    match name.split("__"):
        case ["free", token] if token:
            return None
        case ["slurm", cluster, job, restart, token] if (
            cluster and token and job.isdigit() and restart.isdigit()
        ):
            if slurm.gone(cluster, job):
                return None
            holder = f"Slurm job {job} on cluster {cluster}, which may still run"
        case ["local", host, pid, token] if host and pid and token:
            holder = f"process {pid} on {host}, which furu cannot tell has stopped"
        case _:
            holder = "nothing furu knows of: its name is not one furu gives"
    return (
        f"{run / name} is held by {holder}. If it has stopped, run "
        f"`mv {run / name} {run / 'free__manual'}` to go on from what it left, "
        "or remove the folder to start over"
    )


def held(run: Path) -> str | None:
    """Why this process may not take the working folder in attempts/<run>/ now, if so."""
    for name in entries(run):
        if why := _why_held(run, name):
            return why
    return None


def _take(run: Path, dest: Path) -> bool:
    """Rename the working folder in attempts/<run>/ to ``dest``; False when it has none.

    True does not say the rename won: another worker may have taken the
    folder first, so the caller looks. Raises HeldError when the folder is
    held.
    """
    for _ in range(3):
        # A listing taken during another worker's rename can show both names.
        if len(names := entries(run)) <= 1:
            break
    else:
        raise HeldError(
            f"{run} holds {len(names)} working folders, where it holds one at "
            f"most: {', '.join(names)}. Remove all but the one to go on from"
        )
    if not names:
        return False
    if why := _why_held(run, names[0]):
        raise HeldError(why)
    # Lost to another worker, or lost the reply and NFS sent the rename again.
    with suppress(OSError):
        os.rename(run / names[0], dest)
    return True


def acquire(
    run: Path, published: Callable[[], bool], *, token: str | None = None
) -> Claim | None:
    """Take the working folder in attempts/<run>/ for this process, making one if none.

    None when ``published()`` says the run is published by then. ``token``
    ends the folder's name, so the worker that started this process can find
    the folder when the process dies. Raises HeldError when another worker
    holds the folder, or may.
    """
    mine = run / f"{me()}__{token or new_token()}"
    if not _take(run, mine):
        run.parent.mkdir(parents=True, exist_ok=True)
        seed = run.parent / f".seed__{new_token()}"
        seed.mkdir()
        (seed / mine.name).mkdir()
        # Lost when another worker made attempts/<run>/ first.
        with suppress(OSError):
            os.rename(seed, run)
        shutil.rmtree(seed, ignore_errors=True)
    if entries(run) != [mine.name]:
        raise HeldError(f"another worker took the working folder in {run} first")
    claim = Claim(mine, mine.stat())
    # The worker this process took the folder from, or the one that made the
    # folder before, may have published the run.
    if published():
        try:
            mine.rmdir()
        except OSError:
            release(mine)  # holds what an earlier attempt left; not ours to drop
        with suppress(OSError):
            run.rmdir()
        return None
    return claim


def release(mine: Path) -> None:
    """Give the working folder back to any worker."""
    try:
        os.rename(mine, mine.with_name(f"free__{new_token()}"))
    except OSError as exc:
        if mine.name in list_names(mine.parent):
            logger.warning("could not release %s: %s", mine, exc)


def release_child(run: Path, token: str) -> None:
    """Release what a dead process took with ``token``; its worker has reaped it."""
    for name in entries(run):
        if name.endswith(f"__{token}") and not name.startswith("free__"):
            release(run / name)


def publish(claim: Claim, dest: Path) -> None:
    """Rename the working folder to ``dest`` in runs/, without its scratch/.

    A run that is there is never replaced: this attempt is thrown away instead.
    A rename that may have failed leaves the folder as it was.
    """
    mine = claim.path
    try:
        same = os.path.samestat(mine.stat(), claim.stat)
    except FileNotFoundError:
        same = False
    if not same:
        raise HeldError(f"{mine} was taken from this attempt, which publishes nothing")
    attempts = mine.parent.parent
    trash = attempts / f".trash__{new_token()}"
    with suppress(FileNotFoundError):
        os.rename(scratch_dir_in(mine), trash)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(mine, dest)
    except OSError:
        if dest.name not in list_names(dest.parent):
            with suppress(OSError):
                os.rename(trash, scratch_dir_in(mine))
            raise
        # This rename went through and its reply was lost, or another worker
        # published the run first.
        if mine.name in entries(mine.parent):
            lost = attempts / f".trash__{new_token()}"
            os.rename(mine, lost)
            shutil.rmtree(lost, ignore_errors=True)
    shutil.rmtree(trash, ignore_errors=True)
    with suppress(OSError):
        mine.parent.rmdir()


def discard(run: Path) -> None:
    """Remove the working folder in attempts/<run>/ and its run's folder.

    Raises HeldError when another worker holds the folder, or takes it first.
    """
    trash = run.parent / f".trash__{new_token()}"
    _take(run, trash)
    shutil.rmtree(trash, ignore_errors=True)
    if names := entries(run):
        raise HeldError(f"{run / names[0]} was taken while it was being removed")
    with suppress(OSError):
        run.rmdir()
