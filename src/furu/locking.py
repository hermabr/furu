"""A short exclusive section for appending to schema.ndjson."""

import os
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path


@contextmanager
def short_lock(path: Path) -> Generator[None]:
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            with suppress(FileNotFoundError):
                if time.time() - path.stat().st_mtime > 60:
                    path.unlink()
                    continue
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            with suppress(FileNotFoundError):
                if os.path.samestat(os.fstat(fd), path.stat()):
                    path.unlink()
        finally:
            os.close(fd)
