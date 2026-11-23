"""Minimal short-lived file lock for quick metadata appends (e.g. schema.ndjson)."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

STALE_SECONDS = 60.0
_POLL_SECONDS = 0.05


@contextmanager
def short_lock(
    path: Path,
    *,
    stale_s: float = STALE_SECONDS,
    timeout_s: float = STALE_SECONDS,
    poll_s: float = _POLL_SECONDS,
) -> Generator[None]:
    """Acquire an exclusive lock file, closing the fd before yielding."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            with suppress(OSError):
                if time.time() - path.stat().st_mtime >= stale_s:
                    path.unlink()
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"could not acquire lock at {path}") from None
            time.sleep(min(poll_s, remaining))
    try:
        mine = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        yield
    finally:
        with suppress(OSError):
            if os.path.samestat(mine, path.stat()):
                path.unlink()
