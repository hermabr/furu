import os
import time
from pathlib import Path

import pytest

from furu.locking import LockError, read_text_or_none, short_lock


def test_short_lock_acquires_and_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"

    with short_lock(lock_path):
        assert lock_path.is_file()
        assert f":{os.getpid()}\n" in lock_path.read_text(encoding="utf-8")

    assert not lock_path.exists()


def test_short_lock_times_out_when_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"

    with (
        short_lock(lock_path, stale_s=60.0),
        pytest.raises(LockError, match="could not acquire lock"),
        short_lock(lock_path, stale_s=60.0, timeout_s=0.05, poll_s=0.01),
    ):
        pass


def test_short_lock_breaks_stale_lock_and_does_not_unlink_successors_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "schema.lock"
    lock_path.write_text("stale", encoding="utf-8")
    stale_time = time.time() - 120.0
    os.utime(lock_path, (stale_time, stale_time))

    with short_lock(lock_path, stale_s=60.0, timeout_s=1.0):
        assert lock_path.is_file()
        # Simulate another process replacing the lock after a pause with a new inode.
        other = tmp_path / "other.lock"
        other.write_text("successor", encoding="utf-8")
        other.replace(lock_path)

    assert lock_path.read_text(encoding="utf-8") == "successor"


def test_read_text_or_none(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    assert read_text_or_none(missing) is None
    missing.write_text("hello", encoding="utf-8")
    assert read_text_or_none(missing) == "hello"
