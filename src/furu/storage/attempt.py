"""Who works in a run's folder, said by the folder's name.

``attempts/<run>/`` holds the run's one working folder, under one of:

    free__<token>                                    nobody
    local__<host>__<pid>__<token>                    that process
    slurm__<cluster>__<job id>__<restart>__<token>   a process of that Slurm job

A process takes the folder by renaming it to its own name with a new token.
One rename has one winner on a shared file system, and a listing shows who
that was, so nothing is written, kept alive or timed out. It gives the folder
back as ``free__`` when it ends, and publishing renames the folder out of
``attempts/``. A folder whose process died keeps that process's name, and is
taken only once the name's process is known to be gone. A dotted name is a
folder being deleted.
"""

from __future__ import annotations

import os
import random
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

from furu.utils import listdir_fresh

# How long a process waits for a folder that another one holds.
WAIT_S = 30.0
# The folder everyone makes who finds a run without one, so that they make one.
_SEED = "free__seed"
# What accounting calls a job, or an earlier start of one, whose processes are
# gone. Not NODE_FAIL or BOOT_FAIL, where the node may be running them still,
# nor anything else Slurm may say, such as "Kill task failed".
_ENDED = frozenset(
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
_SQUEUE_EVERY_S = 10.0
_squeue: tuple[float, frozenset[str]] = (float("-inf"), frozenset())
_cluster_name: str | None = None


class Held(RuntimeError):
    """A run's working folder could not be taken."""


def _token() -> str:
    return secrets.token_hex(8)


def _slurm(*argv: str) -> str:
    from furu.worker.backends.slurm.pool import _slurm

    return _slurm(*argv)


def _cluster() -> str | None:
    """The Slurm cluster this machine belongs to, if it can say."""
    global _cluster_name
    if _cluster_name is None:
        _cluster_name = os.environ.get("SLURM_CLUSTER_NAME")
    if _cluster_name is None:
        with suppress(OSError, subprocess.SubprocessError):
            for line in _slurm("scontrol", "show", "config").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "ClusterName":
                    _cluster_name = value.strip()
    return _cluster_name


def _owner() -> str:
    job = os.environ.get("SLURM_JOB_ID")
    if job and (cluster := _cluster()):
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        return f"slurm__{cluster}__{job}__{restart}"
    return f"local__{socket.gethostname()}__{os.getpid()}"


def _queued() -> frozenset[str]:
    """The jobs squeue lists: queued, running or ending.

    Asked at most every 10 s, since a manager looks at many folders. A job that
    started since is unknown to accounting or running there, so it stays held.
    """
    global _squeue
    if time.monotonic() - _squeue[0] > _SQUEUE_EVERY_S:
        jobs = _slurm("squeue", "--noheader", "--format=%A").split()
        _squeue = (time.monotonic(), frozenset(jobs))
    return _squeue[1]


def _why_job_holds(cluster: str, job: str) -> str | None:
    """Why a Slurm job may still run; None when Slurm says twice that it ended."""
    if cluster != _cluster():
        return f"Slurm job {job} of cluster {cluster!r}, which is not this cluster"
    try:
        if job in _queued():
            return f"Slurm job {job}, which is queued or running"
        # Every start of the job, one row each: a job that was requeued ran before.
        rows = _slurm(
            "sacct", "-D", "-X", "--noheader", "--parsable2", "-o", "State", "-j", job
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Slurm job {job}, which Slurm could not be asked about ({exc})"
    # "CANCELLED by <uid>" is CANCELLED.
    states = {row.partition(" by ")[0].strip() for row in rows}
    if not states:
        return f"Slurm job {job}, which accounting has no row for"
    if states - _ENDED:
        return (
            f"Slurm job {job}, which accounting calls "
            f"{', '.join(sorted(states - _ENDED))}"
        )
    return None


def _why_held(name: str) -> str | None:
    """Who may still work in the folder ``name``; None when nobody does."""
    match name.split("__"):
        case ["free", _]:
            return None
        case ["local", host, pid, _] if pid.isdigit():
            if host != socket.gethostname():
                return f"process {pid} on {host}, which only {host} can look for"
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                return None
            except PermissionError:
                pass
            return f"process {pid} on this machine, which is running"
        case ["slurm", cluster, job, _, _]:
            return _why_job_holds(cluster, job)
    return "nobody furu knows: it did not name the folder"


def _visible(names: list[str]) -> list[str]:
    return sorted(name for name in names if not name.startswith("."))


def _why_any_held(parent: Path, names: list[str]) -> str | None:
    for name in names:
        if why := _why_held(name):
            return (
                f"{parent / name} is held by {why}. If nothing works in it, "
                f"rename it to {parent / 'free__0'} and run again."
            )
    return None


def why_held(parent: Path) -> str | None:
    """Why a run's working folder cannot be taken now; None when it can."""
    return _why_any_held(parent, _visible(listdir_fresh(parent)))


def current(parent: Path) -> Path:
    """A run's working folder under the name it has now; ``parent`` when it has none."""
    with suppress(FileNotFoundError):
        if names := _visible(os.listdir(parent)):
            return parent / names[0]
    return parent


def discard(folder: Path, parent: Path) -> None:
    """Delete ``folder`` in one step: under a dotted name in ``parent``, then removed.

    ``parent`` is the ``attempts/<run>/`` of a run whose folder the caller
    holds. Killed in the middle, it leaves the dotted name, which the next
    process to take the run's folder removes.
    """
    trash = parent / f".trash__{_token()}"
    try:
        os.rename(folder, trash)
    except FileNotFoundError:
        return
    shutil.rmtree(trash, ignore_errors=True)


def _release(folder: Path) -> None:
    with suppress(OSError):  # not empty: there is work to go on from
        os.rmdir(folder)
    with suppress(OSError):  # removed just now, or published
        os.rename(folder, folder.with_name(f"free__{_token()}"))
    with suppress(OSError):  # not empty: the run still has its folder
        os.rmdir(folder.parent)


def _try_claim(parent: Path) -> Path | str:
    """Take the run's working folder, making it if the run has none; or say why not."""
    # Made first: a machine that remembers the folder as missing lists it anew.
    os.makedirs(parent, exist_ok=True)
    names = _visible(listdir_fresh(parent))
    if not names:
        # Removed again when its run was published a moment ago.
        with suppress(FileNotFoundError, FileExistsError):
            os.mkdir(parent / _SEED)
        names = _visible(listdir_fresh(parent))
    if why := _why_any_held(parent, names):
        return why
    if len(names) != 1:
        # A second folder was made by a process that listed none a moment
        # before, so it is empty, and only an empty folder is removed here.
        for name in names:
            with suppress(OSError):
                os.rmdir(parent / name)
        return f"{parent} has {len(names)} folders where a run has one"
    mine = parent / f"{_owner()}__{_token()}"
    # Lost to another process, or done with the answer lost: the listing says.
    with suppress(OSError):
        os.rename(parent / names[0], mine)
    now = _visible(listdir_fresh(parent))
    if now == [mine.name]:
        return mine
    if mine.name in now:
        _release(mine)
    return f"{parent} was taken by another process"


@contextmanager
def claimed(parent: Path) -> Generator[Path]:
    """Hold the working folder of the run whose ``attempts/<run>/`` is ``parent``.

    Waits ``WAIT_S`` for a folder that is held, then raises ``Held``. No second
    folder is left beside one that is held, unknown or misnamed.
    """
    deadline = time.monotonic() + WAIT_S
    delay = 0.01
    while isinstance(folder := _try_claim(parent), str):
        if time.monotonic() > deadline:
            raise Held(folder)
        time.sleep(delay * (1 + random.random()))
        delay = min(delay * 2, 5.0)
    # Left by a holder that was killed while it deleted them.
    for name in os.listdir(parent):
        if name.startswith("."):
            shutil.rmtree(parent / name, ignore_errors=True)
    try:
        yield folder
    finally:
        _release(folder)
