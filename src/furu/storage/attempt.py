"""A run's only writable folder is named for its owner. All handoffs are renames."""

import errno
import os
import re
import secrets
import shutil
import socket
import stat
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory

from furu.storage._layout import listed_names, log_path_in
from furu.worker.backends.slurm.commands import clean_exit

_TOKEN = r"[0-9a-f]{16}"
_COMPONENT = r"[a-zA-Z0-9.-]+(?:_[a-zA-Z0-9.-]+)*"
_FREE = re.compile(rf"free__(?:seed|{_TOKEN})")
_SLURM = re.compile(rf"slurm__({_COMPONENT})__([0-9]+)__([0-9]+)__{_TOKEN}")
_LOCAL = re.compile(rf"local__{_COMPONENT}__[0-9]+__{_TOKEN}")
_WAIT_SECONDS = 45


def _owner() -> str:
    token = secrets.token_hex(8)
    if (job := os.environ.get("SLURM_JOB_ID")) is not None:
        cluster = os.environ.get("SLURM_CLUSTER_NAME", "")
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        name = f"slurm__{cluster}__{job}__{restart}__{token}"
        if not _SLURM.fullmatch(name):
            raise RuntimeError(f"Invalid Slurm attempt owner {name!r}")
    else:
        name = f"local__{socket.gethostname()}__{os.getpid()}__{token}"
        if not _LOCAL.fullmatch(name):
            raise RuntimeError(f"Invalid local attempt owner {name!r}")
    return name


def _entries(parent: Path) -> list[Path]:
    return [parent / name for name in listed_names(parent) if not name.startswith(".")]


def current_attempt(parent: Path) -> Path | None:
    entries = _entries(parent)
    if len(entries) == 1:
        entry = entries[0]
        if any(
            pattern.fullmatch(entry.name) for pattern in (_LOCAL, _SLURM, _FREE)
        ) and _directory(entry):
            return entry
    return None


def attempt_log(parent: Path) -> Path:
    """Before a worker claims its folder, only the logical run path is known."""
    attempt = current_attempt(parent)
    return log_path_in(attempt) if attempt else parent


def _directory(entry: Path) -> bool:
    try:
        return stat.S_ISDIR(entry.lstat().st_mode)
    except FileNotFoundError:
        return False


def _adoptable(entry: Path) -> bool:
    if not _directory(entry):
        return False
    if _FREE.fullmatch(entry.name):
        return True
    match = _SLURM.fullmatch(entry.name)
    return bool(match and clean_exit(*match.groups()))


def held(parent: Path) -> bool:
    entries = _entries(parent)
    return bool(entries) and (len(entries) != 1 or not _adoptable(entries[0]))


def discard(folder: Path) -> None:
    """Remove the folder from its namespace before deleting any contents."""
    trash = folder.with_name(f".trash__{secrets.token_hex(8)}")
    folder.rename(trash)
    shutil.rmtree(trash)


def _seed(parent: Path) -> None:
    # Concurrent empty listings must never mkdir a second free__seed beside
    # an owner. POSIX rename can install this parent only over an empty one.
    with TemporaryDirectory(dir=parent.parent, prefix=".seed__") as temporary:
        seed = Path(temporary)
        (seed / "free__seed").mkdir()
        try:
            os.rename(seed, parent)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # A crash after publication can leave only detached scratch here.
            # Move it out atomically before deleting, so competing cleaners
            # and a new owner can all proceed independently.
            names = listed_names(parent)
            if names and all(
                re.fullmatch(rf"\.trash__{_TOKEN}", name) for name in names
            ):
                for name in names:
                    with suppress(FileNotFoundError):
                        os.rename(parent / name, seed / name)


@contextmanager
def claim_attempt(
    parent: Path, *, published: Callable[[], bool] = lambda: False
) -> Generator[Path | None]:
    """Yield a private path, or None when the run was published while waiting."""
    parent.parent.mkdir(parents=True, exist_ok=True)
    mine: Path | None = None
    deadline = time.monotonic() + _WAIT_SECONDS
    while True:
        if published():
            yield None
            return
        entries = _entries(parent)
        if not entries:
            _seed(parent)
            entries = _entries(parent)
        if mine is not None and entries == [
            mine
        ]:  # a previous rename whose reply/listing lagged
            break
        if len(entries) == 1 and _adoptable(entries[0]):
            mine = parent / _owner()
            try:
                os.rename(entries[0], mine)
            except OSError as exc:
                # An NFS rename may succeed at the server and report failure
                # on retransmission. Only the fresh destination proves success.
                if mine not in _entries(parent) and not isinstance(
                    exc, FileNotFoundError
                ):
                    raise
            if _entries(parent) == [mine]:
                break
        if time.monotonic() >= deadline:
            paths = ", ".join(map(str, entries)) or str(parent)
            raise RuntimeError(
                f"Cannot claim attempt at {paths}. Check the owner. If a local "
                "writer has stopped, rename its folder to free__seed and retry."
            )
        time.sleep(1)
    try:
        # A publisher could have moved its folder between our check and seed.
        yield None if published() else mine
    finally:
        with suppress(FileNotFoundError):  # publication consumed the claimed name
            os.rename(mine, parent / f"free__{secrets.token_hex(8)}")
        try:
            parent.rmdir()
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST):
                raise
