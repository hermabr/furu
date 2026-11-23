"""Serialize the short read/append transaction on schema.ndjson."""

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def short_lock(path: Path) -> Generator[None]:
    deadline = time.monotonic() + 10
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Schema writer holds {path}. If it has stopped, remove {path} and retry."
                ) from None
            time.sleep(0.05)
        else:
            break
    try:
        os.close(fd)
        yield
    finally:
        path.unlink()
