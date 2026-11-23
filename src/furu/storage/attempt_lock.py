"""Folder-name attempt locking for runs in ``attempts/<run>/``.

Invariant: ``attempts/<run>/`` holds at most one entry — the active working
folder, named ``free__<tok>``, ``slurm__<cluster>__<job>__<restart>__<tok>``,
or ``local__<host>__<pid>__<tok>``. All temporary/trash folders
(``.seed__<tok>``, ``.trash__<tok>``) live in ``attempts/`` (``run.parent``),
never inside ``attempts/<run>/``.
"""

from __future__ import annotations

import errno
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
from furu.storage._layout import list_names, scratch_dir_in

logger = get_logger("attempt_lock")

_LIST_RETRIES = 3


class HeldError(RuntimeError):
    """Raised when an attempt folder is held by another owner or cannot be claimed."""


@dataclass(frozen=True, slots=True)
class Claim:
    """A claimed attempt working folder and the stat captured at claim time."""

    path: Path
    stat: os.stat_result


def new_token() -> str:
    return secrets.token_hex(8)


def me() -> str:
    """This process's owner prefix: ``slurm__<cluster>__<job>__<restart>`` or ``local__<host>__<pid>``."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if job.isdigit() and (cluster_name := slurm.cluster()):
        restart = os.environ.get("SLURM_RESTART_COUNT") or "0"
        if not restart.isdigit():
            restart = "0"
        return f"slurm__{cluster_name}__{job}__{restart}"
    return f"local__{socket.gethostname()}__{os.getpid()}"


def entries(run: Path) -> list[str]:
    """Sorted entry names currently inside ``attempts/<run>/``."""
    return sorted(list_names(run))


def _settled_entries(run: Path) -> list[str]:
    """Read ``entries(run)``, retrying briefly if a concurrent rename shows >1 entry."""
    names: list[str] = []
    for _ in range(_LIST_RETRIES):
        names = entries(run)
        if len(names) <= 1:
            return names
    return names


def _why_held(run: Path, name: str) -> str | None:
    """Why ``name`` inside ``run`` cannot be claimed, or None if it can."""
    parts = name.split("__")
    if len(parts) == 2 and parts[0] == "free" and parts[1]:
        return None
    path = run / name
    hint = f"mv {path} {run / 'free__manual'}"
    if (
        len(parts) == 5
        and parts[0] == "slurm"
        and parts[1]
        and parts[2].isdigit()
        and parts[3].isdigit()
        and parts[4]
    ):
        cluster_name, job = parts[1], parts[2]
        if slurm.gone(cluster_name, job):
            return None
        return (
            f"attempt is held by Slurm job {job} on {cluster_name} ({path}); "
            f"if it is no longer running, release it with: {hint}"
        )
    if (
        len(parts) == 4
        and parts[0] == "local"
        and parts[1]
        and parts[2].isdigit()
        and parts[3]
    ):
        return (
            f"attempt is held by local process {parts[2]} on {parts[1]} ({path}); "
            f"if it is no longer running, release it with: {hint}"
        )
    return (
        f"attempt has unrecognized folder {path}; "
        f"if nothing is using it, release it with: {hint}"
    )


def held(run: Path) -> str | None:
    """Return an actionable reason if ``run`` is held, or None if free/absent."""
    names = _settled_entries(run)
    if not names:
        return None
    if len(names) > 1:
        return f"attempt {run} has multiple entries: {', '.join(names)}"
    return _why_held(run, names[0])


def current(run: Path) -> Path | None:
    """The single working folder inside ``run``, or None if absent/ambiguous."""
    names = _settled_entries(run)
    return run / names[0] if len(names) == 1 else None


def _take(run: Path, dest: Path) -> bool:
    """Try to rename the single claimable entry in ``run`` to ``dest``.

    Returns False if ``run`` is absent/empty, True if a rename was attempted.
    Raises HeldError if the entry is held or ``run`` has multiple entries.
    """
    names = _settled_entries(run)
    if len(names) > 1:
        raise HeldError(f"attempt {run} has multiple entries: {', '.join(names)}")
    if not names:
        return False
    if why := _why_held(run, names[0]):
        raise HeldError(why)
    with suppress(OSError):
        os.rename(run / names[0], dest)
    return True


def _not_claimed_error(run: Path, found: list[str]) -> HeldError:
    if len(found) == 1 and (why := _why_held(run, found[0])):
        return HeldError(why)
    return HeldError(f"could not claim {run} (saw {found or 'nothing'})")


def acquire(
    run: Path,
    published: Callable[[], bool],
    *,
    token: str | None = None,
) -> Claim | None:
    """Claim ``attempts/<run>/`` for this process, or return None if already published."""
    mine = run / f"{me()}__{token or new_token()}"
    if not _take(run, mine):
        run.parent.mkdir(parents=True, exist_ok=True)
        seed = run.parent / f".seed__{new_token()}"
        seed.mkdir()
        try:
            (seed / mine.name).mkdir()
            with suppress(OSError):
                os.rename(seed, run)
        finally:
            shutil.rmtree(seed, ignore_errors=True)

    found = _settled_entries(run)
    if found != [mine.name]:
        raise _not_claimed_error(run, found)

    try:
        claimed_stat = mine.stat()
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            raise HeldError(f"lost claim on {run} immediately after acquiring") from exc
        raise

    def _undo_claim() -> None:
        try:
            mine.rmdir()
        except OSError:
            release(mine)
        with suppress(OSError):
            run.rmdir()

    try:
        is_published = published()
    except BaseException:
        _undo_claim()
        raise

    if is_published:
        _undo_claim()
        return None
    return Claim(mine, claimed_stat)


def release(mine: Path) -> None:
    """Rename a claimed working folder back to ``free__<tok>``."""
    with suppress(OSError):
        os.rename(mine, mine.with_name(f"free__{new_token()}"))
        return
    if mine.name in entries(mine.parent):
        logger.warning("could not release attempt folder %s", mine)


def release_child(run: Path, token: str) -> None:
    """Release the claim held by a killed worker child for ``token``."""
    suffix = f"__{token}"
    for name in entries(run):
        if name.endswith(suffix) and not name.startswith("free__"):
            release(run / name)


def publish(mine: Path, dest: Path, claimed_stat: os.stat_result) -> None:
    """Atomically publish ``mine`` into ``dest``. Must be called after log scope closes."""
    try:
        current_stat = mine.stat()
    except OSError as exc:
        raise HeldError(f"lost attempt folder {mine} before publishing") from exc
    if not os.path.samestat(current_stat, claimed_stat):
        raise HeldError(
            f"attempt folder {mine} was replaced before publishing (inode changed)"
        )

    attempts_dir = mine.parent.parent
    trash = attempts_dir / f".trash__{new_token()}"
    moved_scratch = False
    try:
        os.rename(scratch_dir_in(mine), trash)
        moved_scratch = True
    except FileNotFoundError:
        pass

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(mine, dest)
    except OSError:
        if dest.name in list_names(dest.parent):
            if mine.name in entries(mine.parent):
                loser = attempts_dir / f".trash__{new_token()}"
                with suppress(OSError):
                    os.rename(mine, loser)
                shutil.rmtree(loser, ignore_errors=True)
        else:
            if moved_scratch:
                with suppress(OSError):
                    os.rename(trash, scratch_dir_in(mine))
            raise
    finally:
        if moved_scratch and trash.exists():
            shutil.rmtree(trash, ignore_errors=True)

    with suppress(OSError):
        mine.parent.rmdir()


def discard(run: Path) -> bool:
    """Remove an unheld working folder inside ``run``.

    Returns True if a working folder was discarded, False if none existed.
    Raises HeldError if the folder is held or claimed concurrently.
    """
    trash = run.parent / f".trash__{new_token()}"
    if not _take(run, trash):
        with suppress(OSError):
            run.rmdir()
        return False
    if found := _settled_entries(run):
        raise _not_claimed_error(run, found)
    existed = trash.exists()
    if existed:
        shutil.rmtree(trash, ignore_errors=True)
    with suppress(OSError):
        run.rmdir()
    return existed
