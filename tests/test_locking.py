import errno
import multiprocessing
import os
import signal
import time
from contextlib import suppress
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import furu.locking as locking_module
from furu.locking import (
    LockAcquireError,
    LockLostError,
    NotLockedError,
    StaleLockRaceError,
    lock,
)

# TODO: make this not using vibes, but by actually thinking through the tests

TEST_TIMING_SCALE = 4.0 if os.environ.get("GITHUB_ACTIONS") == "true" else 1.0
TEST_CLOCK_SLOP_S = 0.02
SHORT_SLEEP_S = 0.05 * TEST_TIMING_SCALE
SHORT_LIFETIME_S = 0.05 * TEST_TIMING_SCALE
SHORT_HEARTBEAT_INTERVAL_S = 0.02 * TEST_TIMING_SCALE
PROCESS_TIMEOUT_S = 0.5 * TEST_TIMING_SCALE


@pytest.fixture(autouse=True)
def _small_clock_slop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locking_module, "CLOCK_SLOP_S", TEST_CLOCK_SLOP_S)


def _child_hold_lock(
    lock_path: Path,
    queue: Queue,
    *,
    sleep_s: float = SHORT_SLEEP_S,
    lifetime_s: float = SHORT_LIFETIME_S,
    heartbeat_interval_s: float = SHORT_HEARTBEAT_INTERVAL_S,
    keep: bool = False,
) -> None:
    with suppress(NotLockedError):
        with lock(
            lock_path,
            lifetime_s=lifetime_s,
            heartbeat_interval_s=heartbeat_interval_s,
        ):
            queue.put(True)
            time.sleep(sleep_s)
            queue.put(True)
            if keep:
                queue.get()


def _child_acquire_then_exit(
    lock_path: Path, owner_path: str, *, lifetime_s: float = SHORT_LIFETIME_S
) -> None:
    with lock(
        lock_path,
        lifetime_s=lifetime_s,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        owner = Path(lock_path).read_text(encoding="utf-8").strip()
        Path(owner_path).write_text(owner, encoding="utf-8")
        os._exit(0)


def _child_hold_lock_and_report_heartbeat(
    lock_path: Path,
    pid_queue: Queue,
    release_queue: Queue,
    *,
    lifetime_s: float = SHORT_LIFETIME_S,
    heartbeat_interval_s: float = SHORT_HEARTBEAT_INTERVAL_S,
) -> None:
    with suppress(NotLockedError, LockLostError):
        with lock(
            lock_path,
            lifetime_s=lifetime_s,
            heartbeat_interval_s=heartbeat_interval_s,
        ):
            heartbeat_children = multiprocessing.active_children()
            assert len(heartbeat_children) == 1
            pid_queue.put((os.getpid(), heartbeat_children[0].pid))
            release_queue.get()


def _child_exit_immediately() -> None:
    return


def _dead_pid() -> int:
    proc = Process(target=_child_exit_immediately)
    proc.start()
    proc.join(timeout=PROCESS_TIMEOUT_S)
    assert proc.exitcode == 0
    assert proc.pid is not None
    return proc.pid


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

    with lock(
        lock_path,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()
        time.sleep(SHORT_LIFETIME_S * 4)

        with pytest.raises(LockAcquireError):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass


def test_lock_uses_default_arguments(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with lock(lock_path) as has_lock:
        assert has_lock()


def _drop_current_lock(lock_path: Path) -> None:
    owner_claim_path = Path(lock_path.read_text(encoding="utf-8").strip())
    lock_path.unlink()
    owner_claim_path.unlink()


def test_has_lock_returns_false_when_lock_is_lost(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with pytest.raises(LockLostError, match="lost lock"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ) as has_lock:
            _drop_current_lock(lock_path)
            assert not has_lock()


def test_exit_raises_lock_lost_error_when_lock_is_lost_mid_block(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "test.lck"

    with pytest.raises(LockLostError, match="lost lock"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            _drop_current_lock(lock_path)
            time.sleep(SHORT_LIFETIME_S)


def test_heartbeat_signal_does_not_release_live_parent_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    pid_queue: Queue = Queue()
    release_queue: Queue = Queue()
    proc = Process(
        target=_child_hold_lock_and_report_heartbeat,
        args=(lock_path, pid_queue, release_queue),
    )
    proc.start()

    try:
        _, heartbeat_pid = pid_queue.get(timeout=PROCESS_TIMEOUT_S)
        os.kill(heartbeat_pid, signal.SIGTERM)
        time.sleep(SHORT_SLEEP_S)

        with pytest.raises(LockAcquireError):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass
    finally:
        release_queue.put(True)
        proc.join(timeout=PROCESS_TIMEOUT_S)


def test_try_release_if_parent_dead_releases_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    owner_claim_path = tmp_path / "test.lck.owner.claim"
    owner_claim_path.write_text(str(owner_claim_path), encoding="utf-8")
    locking_module._touch_future(owner_claim_path, lifetime_s=SHORT_LIFETIME_S)
    assert locking_module._try_acquire_lock(
        lock_path=lock_path, owner_claim_path=owner_claim_path
    )

    assert locking_module._try_release_if_parent_dead(
        lock_path=lock_path,
        owner_claim_path=owner_claim_path,
        parent_pid=_dead_pid(),
    )
    assert not lock_path.exists()
    assert not owner_claim_path.exists()


def test_timeout_when_lock_is_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    holder_queue: Queue = Queue()
    holder = Process(
        target=_child_hold_lock,
        args=(lock_path, holder_queue),
        kwargs={"sleep_s": SHORT_SLEEP_S * 4, "lifetime_s": SHORT_LIFETIME_S * 4},
    )
    holder.start()

    try:
        holder_queue.get(timeout=PROCESS_TIMEOUT_S)
        with pytest.raises(LockAcquireError):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass
    finally:
        holder.join(timeout=PROCESS_TIMEOUT_S)


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
        queue.get(timeout=PROCESS_TIMEOUT_S)
        queue.get(timeout=PROCESS_TIMEOUT_S)
        stale = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
        os.utime(lock_path, (stale, stale))
        first_owner = lock_path.read_text(encoding="utf-8").strip()

        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            second_owner = lock_path.read_text(encoding="utf-8").strip()
            assert second_owner != first_owner
    finally:
        queue.put(True)
        proc.join(timeout=PROCESS_TIMEOUT_S)


def test_lock_raises_if_stale_break_would_remove_reacquired_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "test.lck"
    first_owner_claim_path = tmp_path / "test.lck.first.claim"
    second_owner_claim_path = tmp_path / "test.lck.second.claim"

    first_owner_claim_path.write_text(str(first_owner_claim_path), encoding="utf-8")
    locking_module._touch_future(first_owner_claim_path, lifetime_s=SHORT_LIFETIME_S)
    assert locking_module._try_acquire_lock(
        lock_path=lock_path, owner_claim_path=first_owner_claim_path
    )

    stale = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (stale, stale))

    original_try_acquire_stale_break_dir = locking_module._try_acquire_stale_break_dir

    def reacquire_before_break(*, lock_path: Path, lifetime_s: float) -> Path | None:
        _drop_current_lock(lock_path)
        second_owner_claim_path.write_text(
            str(second_owner_claim_path), encoding="utf-8"
        )
        locking_module._touch_future(second_owner_claim_path, lifetime_s=lifetime_s)
        assert locking_module._try_acquire_lock(
            lock_path=lock_path, owner_claim_path=second_owner_claim_path
        )
        return original_try_acquire_stale_break_dir(
            lock_path=lock_path, lifetime_s=lifetime_s
        )

    with patch.object(
        locking_module,
        "_try_acquire_stale_break_dir",
        side_effect=reacquire_before_break,
    ):
        with pytest.raises(StaleLockRaceError, match="changed owners"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass

    assert locking_module._is_owner(
        lock_path=lock_path, owner_claim_path=second_owner_claim_path
    )


def test_preserves_unrelated_existing_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    lock_path.write_text("save me", encoding="utf-8")
    past = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (past, past))

    with pytest.raises(LockAcquireError):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            pass

    assert lock_path.read_text(encoding="utf-8") == "save me"


def test_does_not_break_lock_within_clock_slop(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with lock(
        lock_path,
        lifetime_s=SHORT_LIFETIME_S * 4,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        almost_stale = time.time() - locking_module.CLOCK_SLOP_S + SHORT_SLEEP_S / 2
        os.utime(lock_path, (almost_stale, almost_stale))

        with pytest.raises(LockAcquireError):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
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
        queue.get(timeout=PROCESS_TIMEOUT_S)
        queue.get(timeout=PROCESS_TIMEOUT_S)
        stale = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
        os.utime(lock_path, (stale, stale))
        with patch("os.utime", side_effect=flaky_utime):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                assert lock_path.exists()
    finally:
        queue.put(True)
        proc.join(timeout=PROCESS_TIMEOUT_S)


def test_lock_retries_once_after_benign_link_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    original_link = os.link
    call_count = 0

    def flaky_link(src, dst, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError(errno.ENOENT, "simulated benign race")
        return original_link(src, dst, *args, **kwargs)

    with patch("os.link", side_effect=flaky_link):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            assert lock_path.exists()

    assert call_count == 2


def test_lock_raises_unexpected_link_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with patch("os.link", side_effect=OSError(errno.EINVAL, "bad link")):
        with pytest.raises(OSError, match="bad link"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass


def test_release_raises_lock_lost_for_estale_unlink_error(tmp_path: Path) -> None:
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
        with pytest.raises(LockLostError, match="lost lock"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                assert lock_path.exists()

    assert call_count >= 1


def test_release_raises_unexpected_unlink_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    original_unlink = os.unlink

    def bad_unlink(path, *args, **kwargs):
        if Path(os.fsdecode(path)) == lock_path:
            raise OSError(errno.EINVAL, "bad unlink")
        return original_unlink(path, *args, **kwargs)

    with patch("os.unlink", side_effect=bad_unlink):
        with pytest.raises(OSError, match="bad unlink"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass


def test_stale_check_raises_unexpected_stat_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    queue: Queue = Queue()
    proc = Process(target=_child_hold_lock, args=(lock_path, queue))
    proc.start()

    try:
        queue.get(timeout=PROCESS_TIMEOUT_S)
        with patch("os.stat", side_effect=_RaiseOnMtimeStat()):
            with pytest.raises(OSError, match="st_mtime failure"):
                with lock(
                    lock_path,
                    lifetime_s=SHORT_LIFETIME_S,
                    heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                ):
                    pass
    finally:
        proc.join(timeout=PROCESS_TIMEOUT_S)


def test_linkcount_check_raises_unexpected_stat_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"

    with patch("os.stat", side_effect=_RaiseOnNlinkStat()):
        with pytest.raises(OSError, match="st_nlink failure"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
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
        queue.get(timeout=PROCESS_TIMEOUT_S)
        queue.get(timeout=PROCESS_TIMEOUT_S)
        stale = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
        os.utime(lock_path, (stale, stale))
        with patch("os.unlink", side_effect=flaky_unlink):
            with pytest.raises(OSError, match="bad claim unlink"):
                with lock(
                    lock_path,
                    lifetime_s=SHORT_LIFETIME_S,
                    heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                ):
                    pass
    finally:
        queue.put(True)
        proc.join(timeout=PROCESS_TIMEOUT_S)


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
        proc.join(timeout=PROCESS_TIMEOUT_S)
        assert proc.exitcode == 0
        first_owner = owner_path.read_text(encoding="utf-8").strip()
        assert lock_path.exists()

        with pytest.raises(LockAcquireError):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass

        stale = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
        os.utime(lock_path, (stale, stale))

        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            second_owner = lock_path.read_text(encoding="utf-8").strip()
            assert second_owner != first_owner
    finally:
        proc.join(timeout=PROCESS_TIMEOUT_S)


def test_does_not_break_corrupt_claim_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    claim_path = tmp_path / "test.lck.example.invalid.claim"
    claim_path.write_text("corrupt", encoding="utf-8")
    os.link(claim_path, lock_path)
    past = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (past, past))

    with pytest.raises(LockAcquireError):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            pass

    assert lock_path.exists()
