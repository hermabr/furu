from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def short_lock(path: Path, *, timeout_s: float = 30.0) -> Generator[None]:
    """Hold the file ``path`` for a few file operations, such as one append.

    Nothing takes the lock from a holder that was killed: it only happens
    inside those few operations, and whoever waits then says which file to
    delete.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"{path} has been held for {timeout_s:g} seconds by a process "
                    "that needs it for milliseconds, so that process was "
                    "killed. Delete the file and run again."
                ) from None
            time.sleep(0.01)
    try:
        yield
    finally:
        os.unlink(path)
