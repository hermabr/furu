from __future__ import annotations

import os
import secrets
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

# A holder does a few seconds' work, so a lock this old was left by one that
# was killed.
STALE_S = 60.0
_POLL_S = 0.05


@contextmanager
def short_lock(path: Path) -> Generator[None]:
    """Hold the lock file ``path`` for a few seconds' work.

    Its writes must be safe to repeat: a holder that takes longer than
    STALE_S loses the lock to the next.
    """
    mine = secrets.token_hex(8)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            with suppress(FileNotFoundError):
                if path.stat().st_mtime + STALE_S < time.time():
                    path.unlink()
            time.sleep(_POLL_S)
            continue
        # Closed before the work, so a holder that is killed leaves no open
        # file for NFS to keep under another name.
        os.write(fd, mine.encode())
        os.close(fd)
        break
    try:
        yield
    finally:
        # Not the next holder's: a file made after this one was removed may
        # get its inode number.
        with suppress(FileNotFoundError):
            if path.read_text() == mine:
                path.unlink()
