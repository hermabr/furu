from __future__ import annotations

import errno
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

STALE_S = 60.0
POLL_S = 0.05


class LockError(RuntimeError):
    """A furu lock could not be acquired or is held by another owner."""


@contextmanager
def short_lock(
    path: Path,
    *,
    stale_s: float = STALE_S,
    timeout_s: float | None = None,
    poll_s: float = POLL_S,
) -> Generator[None]:
    """Brief O_CREAT | O_EXCL lock for short metadata writes (e.g. schema.ndjson)."""
    if timeout_s is None:
        timeout_s = stale_s + 10.0
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError as exc:
                if exc.errno in (errno.ENOENT, errno.ESTALE):
                    continue
                raise
            if age >= stale_s:
                try:
                    path.unlink()
                except OSError as exc:
                    if exc.errno not in (errno.ENOENT, errno.ESTALE):
                        raise
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockError(
                    f"could not acquire lock at {path} within {timeout_s:g} seconds"
                ) from None
            time.sleep(min(poll_s, remaining))
        else:
            os.close(fd)
            break
    try:
        yield
    finally:
        try:
            path.unlink()
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ESTALE):
                raise
