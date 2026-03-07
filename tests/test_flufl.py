import errno
import os
import time
from contextlib import suppress
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furu.flufl import CLOCK_SLOP_S, NotLockedError, TimeOutError, lock


def _child_hold_lock(
    lock_path: str,
    queue: Queue,
    *,
    sleep_s: float = 3,
    lifetime_s: float = 15,
    keep: bool = False,
) -> None:
    with suppress(NotLockedError):
        with lock(lock_path, lifetime_s=lifetime_s, timeout_s=0):
            queue.put(True)
            time.sleep(sleep_s)
            queue.put(True)
            if keep:
                queue.get()


def test_refresh_extends_expiration(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with lock(lock_path, lifetime_s=5, timeout_s=0) as refresh:
        expiration_before = lock_path.stat().st_mtime
        time.sleep(0.01)
        refresh()
        assert lock_path.stat().st_mtime > expiration_before


def test_timeout_when_lock_is_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(target=_child_hold_lock, args=(str(lock_path), queue))
    proc.start()

    try:
        queue.get(timeout=1)
        with pytest.raises(TimeOutError):
            with lock(lock_path, lifetime_s=15, timeout_s=0.1):
                pass
    finally:
        proc.join(timeout=5)


def test_break_stale_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(
        target=_child_hold_lock,
        args=(str(lock_path), queue),
        kwargs={"sleep_s": 5, "lifetime_s": 1, "keep": True},
    )
    proc.start()

    try:
        queue.get(timeout=1)
        queue.get(timeout=6)
        stale = time.time() - CLOCK_SLOP_S - 5
        os.utime(lock_path, (stale, stale))
        first_owner = lock_path.read_text(encoding="utf-8").strip()

        with lock(lock_path, lifetime_s=1, timeout_s=1):
            second_owner = lock_path.read_text(encoding="utf-8").strip()
            assert second_owner != first_owner
    finally:
        queue.put(True)
        proc.join(timeout=5)


def test_preserves_unrelated_existing_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    lock_path.write_text("save me", encoding="utf-8")
    past = time.time() - 10
    os.utime(lock_path, (past, past))

    with pytest.raises(TimeOutError):
        with lock(lock_path, lifetime_s=1, timeout_s=0.1):
            pass

    assert lock_path.read_text(encoding="utf-8") == "save me"


def test_does_not_break_lock_within_clock_slop(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with lock(lock_path, lifetime_s=1, timeout_s=0):
        almost_stale = time.time() - CLOCK_SLOP_S + 0.5
        os.utime(lock_path, (almost_stale, almost_stale))

        with pytest.raises(TimeOutError):
            with lock(lock_path, lifetime_s=1, timeout_s=0.1):
                pass


def test_break_stale_lock_ignores_permission_error_touch(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(
        target=_child_hold_lock,
        args=(str(lock_path), queue),
        kwargs={"sleep_s": 5, "lifetime_s": 1, "keep": True},
    )
    proc.start()

    original_utime = os.utime

    def flaky_utime(path: Any, *args, **kwargs):
        if Path(os.fsdecode(path)) == lock_path:
            raise PermissionError("no permission to touch stale lock")
        return original_utime(path, *args, **kwargs)

    try:
        queue.get(timeout=1)
        queue.get(timeout=6)
        stale = time.time() - CLOCK_SLOP_S - 5
        os.utime(lock_path, (stale, stale))
        with patch("os.utime", side_effect=flaky_utime):
            with lock(lock_path, lifetime_s=1, timeout_s=1):
                assert lock_path.exists()
    finally:
        queue.put(True)
        proc.join(timeout=5)


def test_lock_retries_benign_link_errors(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    original_link = os.link
    call_count = 0

    def flaky_link(src, dst, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError(errno.ENOENT, "simulated benign race")
        if call_count == 2:
            raise OSError(errno.ESTALE, "simulated benign race")
        return original_link(src, dst, *args, **kwargs)

    with patch("os.link", side_effect=flaky_link):
        with lock(lock_path, lifetime_s=5, timeout_s=1):
            assert lock_path.exists()

    assert call_count == 3


def test_lock_raises_unexpected_link_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with patch("os.link", side_effect=OSError(errno.EINVAL, "bad link")):
        with pytest.raises(OSError, match="bad link"):
            with lock(lock_path, lifetime_s=5, timeout_s=1):
                pass


def test_does_not_break_corrupt_claim_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    claim_path = tmp_path / "test.lck.example.invalid.claim"
    claim_path.write_text("corrupt", encoding="utf-8")
    os.link(claim_path, lock_path)
    past = time.time() - CLOCK_SLOP_S - 5
    os.utime(lock_path, (past, past))

    with pytest.raises(TimeOutError):
        with lock(lock_path, lifetime_s=1, timeout_s=0.1):
            pass

    assert lock_path.exists()
