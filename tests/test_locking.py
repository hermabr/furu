"""Tests for minimal short_lock()."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from furu.locking import STALE_SECONDS, short_lock


def test_short_lock_creates_and_removes_lock_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"

    with short_lock(lock_path):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_short_lock_closes_fd_before_yielding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "schema.lock"
    real_open = os.open
    real_close = os.close
    open_fds: set[int] = set()

    def track_open(path: Any, flags: int, *args: Any) -> int:
        fd = real_open(path, flags, *args)
        if Path(os.fsdecode(path)) == lock_path:
            open_fds.add(fd)
        return fd

    def track_close(fd: int) -> None:
        open_fds.discard(fd)
        real_close(fd)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", track_close)

    with short_lock(lock_path):
        assert open_fds == set()
        assert lock_path.exists()

    assert not lock_path.exists()


def test_short_lock_refuses_fresh_lock_without_sleeping(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"
    with short_lock(lock_path):
        with (
            pytest.raises(TimeoutError, match="could not acquire lock"),
            short_lock(lock_path, timeout_s=0.0),
        ):
            pass
        assert lock_path.exists()


def test_short_lock_breaks_stale_lock_by_mtime(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"
    lock_path.write_text("stale")
    # Pin the old inode with a hardlink so the kernel cannot immediately recycle
    # the same inode number after short_lock unlinks the stale lock.
    os.link(lock_path, tmp_path / "keep-old-inode")
    old = time.time() - STALE_SECONDS - 5.0
    os.utime(lock_path, (old, old))
    old_stat = lock_path.stat()

    with short_lock(lock_path, timeout_s=0.0):
        assert lock_path.exists()
        assert not os.path.samestat(old_stat, lock_path.stat())
        assert lock_path.read_text() == ""

    assert not lock_path.exists()


def test_short_lock_finally_checks_samestat_before_unlinking(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"
    replacement_stat: os.stat_result | None = None

    with short_lock(lock_path):
        # Allocate a replacement file while lock_path still exists so its inode
        # is guaranteed distinct, then atomically replace lock_path.
        replacement = tmp_path / "replacement.lock"
        replacement.write_text("other-owner")
        replacement_stat = replacement.stat()
        os.replace(replacement, lock_path)

    assert replacement_stat is not None
    assert lock_path.exists()
    assert os.path.samestat(replacement_stat, lock_path.stat())
