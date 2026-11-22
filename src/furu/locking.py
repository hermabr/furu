"""A short lock: one file made with O_EXCL, held for milliseconds."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

# Older than this, a lock file is a killed holder's. A holder frozen for longer
# loses the lock, so what it guards must stay right without it.
STALE_S = 60.0


@contextmanager
def short_lock(path: Path) -> Generator[None]:
    """Hold ``path`` while a shared file gets a few lines appended."""
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            with suppress(FileNotFoundError):
                if time.time() - path.stat().st_mtime > STALE_S:
                    path.unlink()
                    continue
            time.sleep(0.05)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
