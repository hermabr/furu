"""A lock for a moment's work on one file. Only schema.ndjson's writers take it."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

# A holder is done in milliseconds, so a lock this old was left by a killed one.
_STALE_S = 60.0
_POLL_S = 0.05


@contextmanager
def short_lock(path: Path) -> Generator[None]:
    """Hold ``path`` while the body runs: creating the file is taking the lock.

    Nothing keeps the lock alive, so a holder that stops for longer than
    ``_STALE_S`` can find that another took it. What the lock guards has to
    stay right then too.
    """
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            with suppress(FileNotFoundError):  # released since
                if time.time() - path.stat().st_mtime > _STALE_S:
                    path.unlink()
                    continue
            time.sleep(_POLL_S)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
