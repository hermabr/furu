from __future__ import annotations

import errno
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


class LockError(RuntimeError):
    """A furu lock could not be acquired, was lost, or is inconsistent on disk."""


def _is_missing_or_stale(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESTALE)


def read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return None
        raise


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:
        if not _is_missing_or_stale(exc):
            raise


@contextmanager
def short_lock(
    lock_path: Path,
    *,
    stale_after_s: float = 30.0,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.05,
) -> Generator[None]:
    """Short-lived exclusive file lock via O_CREAT | O_EXCL, with stale-mtime breaking."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                st = lock_path.stat()
            except OSError as exc:
                if _is_missing_or_stale(exc):
                    continue
                raise
            if st.st_mtime + stale_after_s <= time.time():
                unlink_if_exists(lock_path)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockError(
                    f"could not acquire lock at {lock_path} within {timeout_s:g} seconds"
                ) from None
            time.sleep(min(poll_interval_s, remaining))
        else:
            os.close(fd)
            break
    try:
        yield
    finally:
        unlink_if_exists(lock_path)
