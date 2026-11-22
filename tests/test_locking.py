from __future__ import annotations

import errno
import os
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from furu.locking import LockError, short_lock


def test_short_lock_creates_and_removes_lock_file_without_heartbeat_thread(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "schema.lock"
    before = {t.name for t in threading.enumerate()}

    with short_lock(lock_path):
        assert lock_path.exists()
        assert {t.name for t in threading.enumerate()} == before

    assert not lock_path.exists()


def test_short_lock_times_out_while_fresh_lock_is_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"

    with (
        short_lock(lock_path),
        pytest.raises(LockError, match="could not acquire lock"),
        short_lock(lock_path, stale_s=60.0, timeout_s=0.05, poll_s=0.01),
    ):
        pass


def test_short_lock_waits_for_release_before_timeout(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"
    holder = ExitStack()
    holder.enter_context(short_lock(lock_path))

    def release_soon() -> None:
        time.sleep(0.03)
        holder.close()

    thread = threading.Thread(target=release_soon)
    thread.start()
    try:
        with short_lock(lock_path, stale_s=60.0, timeout_s=5.0, poll_s=0.01):
            assert lock_path.exists()
    finally:
        thread.join()

    assert not lock_path.exists()


def test_short_lock_breaks_stale_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"
    lock_path.touch()
    old = time.time() - 120.0
    os.utime(lock_path, (old, old))

    with short_lock(lock_path, stale_s=60.0, timeout_s=1.0, poll_s=0.01):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_short_lock_ignores_missing_or_estale_on_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"

    with short_lock(lock_path):
        lock_path.unlink()

    with (
        patch.object(Path, "unlink", side_effect=OSError(errno.ESTALE, "stale")),
        short_lock(lock_path),
    ):
        pass
