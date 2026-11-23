"""A lock for a moment's work on one file. Only schema.ndjson's writers take it."""

from __future__ import annotations

import errno
import os
import secrets
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

# A holder is done in milliseconds, so a lock this old was left by a killed one.
_STALE_S = 60.0
_POLL_S = 0.05


def _missing(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESTALE)


@contextmanager
def short_lock(path: Path) -> Generator[None]:
    """Hold ``path`` while the body runs: creating the file is taking the lock.

    Nothing keeps the lock alive, so a holder that stops for longer than
    ``_STALE_S`` can find that another took it. What the lock guards has to
    stay right then too.
    """
    mine = secrets.token_hex(8)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > _STALE_S:
                    path.unlink()
                    continue
            except OSError as exc:  # released since
                if not _missing(exc):
                    raise
            time.sleep(_POLL_S)
    # Closed before the body: a file removed while it is open stays on the
    # file server under another name. Writing a token identifies this holder
    # even if the filesystem reuses the inode number after a stale break.
    try:
        os.write(fd, mine.encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        # Another's by now if this one was taken for stale.
        try:
            if path.read_text(encoding="utf-8") == mine:
                path.unlink()
        except OSError as exc:
            if not _missing(exc):
                raise
