from __future__ import annotations

import errno
import os
import socket
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path


class LockError(RuntimeError):
    """A furu lock could not be acquired or is inconsistent on disk."""


def read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            return None
        raise


@contextmanager
def short_lock(
    lock_path: Path,
    *,
    stale_s: float = 60.0,
    timeout_s: float = 70.0,
    poll_s: float = 0.05,
) -> Generator[None]:
    """Exclusive O_CREAT | O_EXCL lock for short metadata writes (no heartbeat)."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            with suppress(OSError):
                if lock_path.stat().st_mtime + stale_s <= time.time():
                    lock_path.unlink(missing_ok=True)
                    continue
            if time.monotonic() >= deadline:
                raise LockError(f"could not acquire lock at {lock_path}") from None
            time.sleep(poll_s)
        else:
            try:
                os.write(fd, f"{socket.gethostname()}:{os.getpid()}\n".encode())
                my_stat = os.fstat(fd)
            finally:
                os.close(fd)
            break
    try:
        yield
    finally:
        with suppress(OSError):
            if os.path.samestat(lock_path.stat(), my_stat):
                lock_path.unlink()
