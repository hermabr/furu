import errno
import os
import time
from contextlib import suppress
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import furu.locking_manager as locking_manager_module
from furu.locking_manager import NotLockedError, TimeOutError, lock

TEST_CLOCK_SLOP_S = 0.02
SHORT_SLEEP_S = 0.05
SHORT_LIFETIME_S = 0.05
SHORT_TIMEOUT_S = 0.05


@pytest.fixture(autouse=True)
def _small_clock_slop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locking_manager_module, "CLOCK_SLOP_S", TEST_CLOCK_SLOP_S)


def _child_hold_lock(
    lock_path: Path,
    queue: Queue,
    *,
    sleep_s: float = SHORT_SLEEP_S,
    lifetime_s: float = SHORT_LIFETIME_S,
    keep: bool = False,
) -> None:
    with suppress(NotLockedError):
        with lock(lock_path, lifetime_s=lifetime_s, timeout_s=0):
            queue.put(True)
            time.sleep(sleep_s)
            queue.put(True)
            if keep:
                queue.get()


def _child_acquire_then_exit(
    lock_path: Path, owner_path: str, *, lifetime_s: float = SHORT_LIFETIME_S
) -> None:
    with lock(lock_path, lifetime_s=lifetime_s, timeout_s=0):
        owner = Path(lock_path).read_text(encoding="utf-8").strip()
        Path(owner_path).write_text(owner, encoding="utf-8")
        os._exit(0)


class _FakeMtimeStat:
    def __init__(self, stat_result: os.stat_result):
        self._stat_result = stat_result

    def __getattr__(self, name: str):
        if name == "st_mtime":
            raise OSError(errno.EINVAL, "st_mtime failure")
        return getattr(self._stat_result, name)


class _RaiseOnMtimeStat:
    def __init__(self):
        self._os_stat = os.stat

    def __call__(self, path, *args, **kwargs):
        return _FakeMtimeStat(self._os_stat(path, *args, **kwargs))


class _FakeNlinkStat:
    def __init__(self, stat_result: os.stat_result):
        self._stat_result = stat_result

    def __getattr__(self, name: str):
        if name == "st_nlink":
            raise OSError(errno.EINVAL, "st_nlink failure")
        return getattr(self._stat_result, name)


class _RaiseOnNlinkStat:
    def __init__(self):
        self._os_stat = os.stat

    def __call__(self, path, *args, **kwargs):
        return _FakeNlinkStat(self._os_stat(path, *args, **kwargs))


def test_refresh_extends_expiration(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0):
        owner_claim_path = Path(lock_path.read_text(encoding="utf-8").strip())
        expiration_before = owner_claim_path.stat().st_mtime
        deadline = time.time() + 0.5
        while time.time() < deadline:
            if owner_claim_path.stat().st_mtime > expiration_before:
                break
            time.sleep(0.005)

        assert owner_claim_path.stat().st_mtime > expiration_before


def test_timeout_when_lock_is_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(
        target=_child_hold_lock,
        args=(lock_path, queue),
        kwargs={"sleep_s": SHORT_SLEEP_S * 4, "lifetime_s": SHORT_LIFETIME_S * 4},
    )
    proc.start()

    try:
        queue.get(timeout=0.5)
        with pytest.raises(TimeOutError):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                timeout_s=SHORT_TIMEOUT_S / 2,
            ):
                pass
    finally:
        proc.join(timeout=0.5)


def test_break_stale_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(
        target=_child_hold_lock,
        args=(lock_path, queue),
        kwargs={"sleep_s": SHORT_SLEEP_S, "lifetime_s": SHORT_LIFETIME_S, "keep": True},
    )
    proc.start()

    try:
        queue.get(timeout=0.5)
        queue.get(timeout=0.5)
        stale = time.time() - locking_manager_module.CLOCK_SLOP_S - SHORT_SLEEP_S
        os.utime(lock_path, (stale, stale))
        first_owner = lock_path.read_text(encoding="utf-8").strip()

        with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0.5):
            second_owner = lock_path.read_text(encoding="utf-8").strip()
            assert second_owner != first_owner
    finally:
        queue.put(True)
        proc.join(timeout=0.5)


def test_preserves_unrelated_existing_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    lock_path.write_text("save me", encoding="utf-8")
    past = time.time() - locking_manager_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (past, past))

    with pytest.raises(TimeOutError):
        with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=SHORT_TIMEOUT_S):
            pass

    assert lock_path.read_text(encoding="utf-8") == "save me"


def test_does_not_break_lock_within_clock_slop(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with lock(lock_path, lifetime_s=SHORT_LIFETIME_S * 4, timeout_s=0):
        almost_stale = (
            time.time() - locking_manager_module.CLOCK_SLOP_S + SHORT_TIMEOUT_S / 2
        )
        os.utime(lock_path, (almost_stale, almost_stale))

        with pytest.raises(TimeOutError):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                timeout_s=SHORT_TIMEOUT_S / 4,
                poll_interval_s=SHORT_TIMEOUT_S / 10,
            ):
                pass


def test_break_stale_lock_ignores_permission_error_touch(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(
        target=_child_hold_lock,
        args=(lock_path, queue),
        kwargs={"sleep_s": SHORT_SLEEP_S, "lifetime_s": SHORT_LIFETIME_S, "keep": True},
    )
    proc.start()

    original_utime = os.utime

    def flaky_utime(path: Any, *args, **kwargs):
        if Path(os.fsdecode(path)) == lock_path:
            raise PermissionError("no permission to touch stale lock")
        return original_utime(path, *args, **kwargs)

    try:
        queue.get(timeout=0.5)
        queue.get(timeout=0.5)
        stale = time.time() - locking_manager_module.CLOCK_SLOP_S - SHORT_SLEEP_S
        os.utime(lock_path, (stale, stale))
        with patch("os.utime", side_effect=flaky_utime):
            with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0.5):
                assert lock_path.exists()
    finally:
        queue.put(True)
        proc.join(timeout=0.5)


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
        with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0.5):
            assert lock_path.exists()

    assert call_count == 3


def test_lock_raises_unexpected_link_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with patch("os.link", side_effect=OSError(errno.EINVAL, "bad link")):
        with pytest.raises(OSError, match="bad link"):
            with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0.5):
                pass


def test_release_ignores_estale_unlink_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    original_unlink = os.unlink
    call_count = 0

    def flaky_unlink(path, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError(errno.ESTALE, "stale unlink")
        return original_unlink(path, *args, **kwargs)

    with patch("os.unlink", side_effect=flaky_unlink):
        with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0):
            assert lock_path.exists()

    assert call_count >= 2


def test_release_raises_unexpected_unlink_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    original_unlink = os.unlink

    def bad_unlink(path, *args, **kwargs):
        if Path(os.fsdecode(path)) == lock_path:
            raise OSError(errno.EINVAL, "bad unlink")
        return original_unlink(path, *args, **kwargs)

    with patch("os.unlink", side_effect=bad_unlink):
        with pytest.raises(OSError, match="bad unlink"):
            with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0):
                pass


def test_stale_check_raises_unexpected_stat_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(target=_child_hold_lock, args=(lock_path, queue))
    proc.start()

    try:
        queue.get(timeout=0.5)
        with patch("os.stat", side_effect=_RaiseOnMtimeStat()):
            with pytest.raises(OSError, match="st_mtime failure"):
                with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0.5):
                    pass
    finally:
        proc.join(timeout=0.5)


def test_linkcount_check_raises_unexpected_stat_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with patch("os.stat", side_effect=_RaiseOnNlinkStat()):
        with pytest.raises(OSError, match="st_nlink failure"):
            with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0.5):
                pass


def test_break_stale_lock_raises_if_winner_claim_unlink_fails(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(
        target=_child_hold_lock,
        args=(lock_path, queue),
        kwargs={"sleep_s": SHORT_SLEEP_S, "lifetime_s": SHORT_LIFETIME_S, "keep": True},
    )
    proc.start()

    original_unlink = os.unlink

    def flaky_unlink(path, *args, **kwargs):
        if Path(os.fsdecode(path)) != lock_path:
            raise OSError(errno.EINVAL, "bad claim unlink")
        return original_unlink(path, *args, **kwargs)

    try:
        queue.get(timeout=0.5)
        queue.get(timeout=0.5)
        stale = time.time() - locking_manager_module.CLOCK_SLOP_S - SHORT_SLEEP_S
        os.utime(lock_path, (stale, stale))
        with patch("os.unlink", side_effect=flaky_unlink):
            with pytest.raises(OSError, match="bad claim unlink"):
                with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0.5):
                    pass
    finally:
        queue.put(True)
        proc.join(timeout=0.5)


def test_process_exit_without_cleanup_allows_reclaim_after_expiry(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "test.lck"
    owner_path = tmp_path / "owner.txt"
    proc = Process(
        target=_child_acquire_then_exit,
        args=(lock_path, str(owner_path)),
        kwargs={"lifetime_s": SHORT_LIFETIME_S},
    )
    proc.start()

    try:
        proc.join(timeout=0.5)
        assert proc.exitcode == 0
        first_owner = owner_path.read_text(encoding="utf-8").strip()
        assert lock_path.exists()

        with pytest.raises(TimeOutError):
            with lock(
                lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=SHORT_TIMEOUT_S
            ):
                pass

        stale = time.time() - locking_manager_module.CLOCK_SLOP_S - SHORT_SLEEP_S
        os.utime(lock_path, (stale, stale))

        with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=0.5):
            second_owner = lock_path.read_text(encoding="utf-8").strip()
            assert second_owner != first_owner
    finally:
        proc.join(timeout=0.5)


def test_does_not_break_corrupt_claim_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    claim_path = tmp_path / "test.lck.example.invalid.claim"
    claim_path.write_text("corrupt", encoding="utf-8")
    os.link(claim_path, lock_path)
    past = time.time() - locking_manager_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (past, past))

    with pytest.raises(TimeOutError):
        with lock(lock_path, lifetime_s=SHORT_LIFETIME_S, timeout_s=SHORT_TIMEOUT_S):
            pass

    assert lock_path.exists()
