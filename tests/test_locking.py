import errno
import json
import logging
import os
import threading
import time
from contextlib import suppress
from multiprocessing import Process, Queue
from pathlib import Path
from unittest.mock import patch

import pytest

import furu.locking as locking_module
from furu.locking import (
    LockError,
    LockManifest,
    lock,
)

TEST_TIMING_SCALE = 4.0 if os.environ.get("GITHUB_ACTIONS") == "true" else 1.0
TEST_CLOCK_SLOP_S = 0.02
SHORT_SLEEP_S = 0.05 * TEST_TIMING_SCALE
SHORT_LIFETIME_S = 0.05 * TEST_TIMING_SCALE
SHORT_HEARTBEAT_INTERVAL_S = 0.02 * TEST_TIMING_SCALE
PROCESS_TIMEOUT_S = 0.5 * TEST_TIMING_SCALE


@pytest.fixture(autouse=True)
def _small_clock_slop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locking_module, "CLOCK_SLOP_S", TEST_CLOCK_SLOP_S)


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_claim_path(path: Path) -> Path:
    return Path(str(_read_manifest(path)["claim_path"]))


def _child_hold_lock(
    lock_path: Path,
    queue: Queue,
    *,
    sleep_s: float = SHORT_SLEEP_S,
    lifetime_s: float = SHORT_LIFETIME_S,
    heartbeat_interval_s: float = SHORT_HEARTBEAT_INTERVAL_S,
    keep: bool = False,
) -> None:
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


def _child_acquire_batch_then_exit(lock_paths: list[Path], manifest_out: str) -> None:
    with lock(
        lock_paths,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        Path(manifest_out).write_text(lock_paths[0].read_text(encoding="utf-8"))
        os._exit(0)


def _child_acquire_then_exit(
    lock_path: Path, owner_path: str, *, lifetime_s: float = SHORT_LIFETIME_S
) -> None:
    with lock(
        lock_path,
        lifetime_s=lifetime_s,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        Path(owner_path).write_text(
            str(_manifest_claim_path(lock_path)), encoding="utf-8"
        )
        os._exit(0)


def _child_hold_lock_and_report_heartbeat(
    lock_path: Path,
    thread_queue: Queue,
    release_queue: Queue,
    *,
    lifetime_s: float = SHORT_LIFETIME_S,
    heartbeat_interval_s: float = SHORT_HEARTBEAT_INTERVAL_S,
) -> None:
    with suppress(RuntimeError):
        with lock(
            lock_path,
            lifetime_s=lifetime_s,
            heartbeat_interval_s=heartbeat_interval_s,
        ):
            heartbeat_threads = [
                thread.name
                for thread in threading.enumerate()
                if thread.name.startswith("lock-heartbeat:")
            ]
            assert len(heartbeat_threads) == 1
            thread_queue.put((os.getpid(), heartbeat_threads[0]))
            release_queue.get()


def _drop_current_lock(lock_path: Path) -> None:
    claim_path = _manifest_claim_path(lock_path)
    lock_path.unlink()
    claim_path.unlink()


def test_lock_with_many_paths_produces_hardlinks_to_one_manifest_inode(
    tmp_path: Path,
) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]

    with lock(
        lock_paths,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()

        manifest = _read_manifest(lock_paths[0])

        claim_path = Path(str(manifest["claim_path"]))
        claim_stat = claim_path.stat()
        assert manifest["lock_paths"] == [str(path.resolve()) for path in lock_paths]

        for lock_path in lock_paths:
            assert os.path.samestat(lock_path.stat(), claim_stat)


def test_lock_accepts_a_single_path(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with lock(lock_path, acquire_timeout_s=1.0) as has_lock:
        assert has_lock()

        manifest = _read_manifest(lock_path)
        assert manifest["lock_paths"] == [str(lock_path.resolve())]

    assert not lock_path.exists()


def test_lock_uses_default_arguments(tmp_path: Path) -> None:
    with lock(tmp_path / "single.lock") as has_lock:
        assert has_lock()


def test_lock_normalizes_paths_and_shares_one_claim_manifest(
    tmp_path: Path,
) -> None:
    lock_a = tmp_path / "a.lock"
    lock_b = tmp_path / "b.lock"

    with lock([lock_b, lock_a, lock_a]) as has_lock:
        manifest_a = _read_manifest(lock_a)
        manifest_b = _read_manifest(lock_b)

        assert has_lock()
        assert manifest_a == manifest_b
        assert manifest_a["lock_paths"] == [
            str(lock_a.resolve()),
            str(lock_b.resolve()),
        ]


def test_has_lock_returns_false_when_lock_is_lost(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with pytest.raises(LockError, match="lost lock"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ) as has_lock:
            _drop_current_lock(lock_path)
            assert not has_lock()


def test_has_lock_returns_false_when_any_batch_link_is_lost(tmp_path: Path) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]

    with pytest.raises(LockError, match="lost lock"):
        with lock(
            lock_paths,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ) as has_lock:
            lock_paths[1].unlink()
            assert not has_lock()


def test_exit_raises_lock_lost_error_when_lock_is_lost_mid_block(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"

    with pytest.raises(LockError, match="lost lock"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            _drop_current_lock(lock_path)
            time.sleep(SHORT_LIFETIME_S)


def test_refresh_extends_expiration(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with lock(
        lock_path,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()
        time.sleep(SHORT_LIFETIME_S * 4)

        with pytest.raises(LockError, match="could not acquire lock"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ):
                pass


def test_lock_starts_heartbeat_thread(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"
    thread_queue: Queue = Queue()
    release_queue: Queue = Queue()
    proc = Process(
        target=_child_hold_lock_and_report_heartbeat,
        args=(lock_path, thread_queue, release_queue),
    )
    proc.start()

    try:
        _, heartbeat_name = thread_queue.get(timeout=PROCESS_TIMEOUT_S)
        assert heartbeat_name == f"lock-heartbeat:{lock_path.name}"

        with pytest.raises(LockError, match="could not acquire lock"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ):
                pass
    finally:
        release_queue.put(True)
        proc.join(timeout=PROCESS_TIMEOUT_S)


def test_timeout_when_lock_is_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"
    holder_queue: Queue = Queue()
    holder = Process(
        target=_child_hold_lock,
        args=(lock_path, holder_queue),
        kwargs={"sleep_s": SHORT_SLEEP_S * 4, "lifetime_s": SHORT_LIFETIME_S * 4},
    )
    holder.start()

    try:
        holder_queue.get(timeout=PROCESS_TIMEOUT_S)
        started_at = time.monotonic()
        with pytest.raises(LockError, match="could not acquire lock"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass
        assert time.monotonic() - started_at >= SHORT_LIFETIME_S
    finally:
        holder.join(timeout=PROCESS_TIMEOUT_S)


def test_waits_for_lock_release_before_timeout(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"
    holder_queue: Queue = Queue()
    holder = Process(
        target=_child_hold_lock,
        args=(lock_path, holder_queue),
        kwargs={"sleep_s": SHORT_SLEEP_S * 2, "lifetime_s": SHORT_LIFETIME_S * 8},
    )
    holder.start()

    try:
        holder_queue.get(timeout=PROCESS_TIMEOUT_S)
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            acquire_timeout_s=SHORT_SLEEP_S * 4,
            acquire_poll_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            assert lock_path.exists()
    finally:
        holder.join(timeout=PROCESS_TIMEOUT_S)


def test_lock_logs_when_waiting_for_lock(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"
    furu_logger = logging.getLogger("furu")
    furu_logger.addHandler(caplog.handler)
    monkeypatch.setattr(
        locking_module,
        "DEFAULT_LOCK_WAIT_LOG_INTERVAL_S",
        SHORT_HEARTBEAT_INTERVAL_S / 2,
    )

    try:
        caplog.set_level(logging.INFO, logger="furu")
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S * 4,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            with pytest.raises(LockError, match="could not acquire lock"):
                with lock(
                    lock_path,
                    lifetime_s=SHORT_LIFETIME_S,
                    heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                    acquire_timeout_s=SHORT_SLEEP_S,
                    acquire_poll_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                ):
                    pass
    finally:
        furu_logger.removeHandler(caplog.handler)

    wait_message = f"waiting for lock at {lock_path.resolve()}"
    assert caplog.messages.count(wait_message) >= 2


def test_stale_break_from_any_member_path_removes_whole_logical_lock_group(
    tmp_path: Path,
) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]
    manifest_out = tmp_path / "manifest.json"
    proc = Process(
        target=_child_acquire_batch_then_exit,
        args=(lock_paths, str(manifest_out)),
    )
    proc.start()
    proc.join(timeout=PROCESS_TIMEOUT_S)
    assert proc.exitcode == 0

    stale_manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    stale_claim_path = Path(str(stale_manifest["claim_path"]))
    stale_time = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(stale_claim_path, (stale_time, stale_time))

    with lock(
        lock_paths[1],
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()
        assert not lock_paths[0].exists()
        assert not lock_paths[2].exists()
        assert not stale_claim_path.exists()


def test_stale_break_from_any_member_path_clears_group_when_claim_file_is_missing(
    tmp_path: Path,
) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]
    manifest_out = tmp_path / "manifest.json"
    proc = Process(
        target=_child_acquire_batch_then_exit,
        args=(lock_paths, str(manifest_out)),
    )
    proc.start()
    proc.join(timeout=PROCESS_TIMEOUT_S)
    assert proc.exitcode == 0

    stale_manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    stale_claim_path = Path(str(stale_manifest["claim_path"]))
    stale_claim_path.unlink()

    stale_time = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_paths[0], (stale_time, stale_time))

    with lock(
        lock_paths[1],
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()
        assert not lock_paths[0].exists()
        assert not lock_paths[2].exists()
        assert not stale_claim_path.exists()


def test_release_cleanup_removes_member_links_when_claim_file_is_missing(
    tmp_path: Path,
) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]

    with pytest.raises(LockError, match="lost lock"):
        with lock(
            lock_paths,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            _manifest_claim_path(lock_paths[0]).unlink()

    assert not any(lock_path.exists() for lock_path in lock_paths)


def test_lock_raises_if_stale_break_would_remove_reacquired_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"
    first_claim_path = tmp_path / "first.claim"
    second_claim_path = tmp_path / "second.claim"
    manifest = LockManifest(
        claim_path=first_claim_path.resolve(),
        lock_paths=(lock_path.resolve(),),
    )

    locking_module._write_claim_file(manifest, lifetime_s=SHORT_LIFETIME_S)
    assert locking_module.try_link(
        lock_path=lock_path,
        claim_path=first_claim_path.resolve(),
    )

    stale = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (stale, stale))

    original_acquire_break_file = locking_module.acquire_break_file

    def reacquire_before_break(*, claim_path: Path, lifetime_s: float) -> Path | None:
        break_path = original_acquire_break_file(
            claim_path=claim_path,
            lifetime_s=lifetime_s,
        )
        if break_path is None:
            return None

        lock_path.unlink()
        second_manifest = LockManifest(
            claim_path=second_claim_path.resolve(),
            lock_paths=(lock_path.resolve(),),
        )
        locking_module._write_claim_file(second_manifest, lifetime_s=lifetime_s)
        assert locking_module.try_link(
            lock_path=lock_path,
            claim_path=second_claim_path.resolve(),
        )
        os.utime(lock_path, (stale, stale))
        return break_path

    with patch.object(
        locking_module,
        "acquire_break_file",
        side_effect=reacquire_before_break,
    ):
        with pytest.raises(
            LockError, match="changed owners while breaking a stale lock"
        ):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass

    assert locking_module.owns([lock_path], claim_path=second_claim_path.resolve())


def test_cross_filesystem_lock_requests_raise_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_parent = tmp_path / "one"
    second_parent = tmp_path / "two"
    first_parent.mkdir()
    second_parent.mkdir()
    lock_paths = [first_parent / "a.lock", second_parent / "b.lock"]
    real_stat = Path.stat

    class _FakeStat:
        def __init__(self, stat_result: os.stat_result, *, st_dev: int):
            self._stat_result = stat_result
            self.st_dev = st_dev

        def __getattr__(self, name: str):
            return getattr(self._stat_result, name)

    def fake_stat(path: Path, *args, **kwargs):
        stat_result = real_stat(path, *args, **kwargs)
        if path == lock_paths[1].parent:
            return _FakeStat(stat_result, st_dev=stat_result.st_dev + 1)
        return stat_result

    monkeypatch.setattr(type(lock_paths[0].parent), "stat", fake_stat)

    with pytest.raises(LockError, match="same filesystem device"):
        with lock(lock_paths):
            pass


def test_partial_acquire_rollback_releases_subset_under_overlap(tmp_path: Path) -> None:
    first_lock = tmp_path / "a.lock"
    blocked_lock = tmp_path / "b.lock"

    with lock(
        blocked_lock,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        with pytest.raises(LockError, match="could not acquire lock"):
            with lock(
                [first_lock, blocked_lock],
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ):
                pass

    assert not first_lock.exists()


def test_does_not_break_lock_within_clock_slop(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with lock(
        lock_path,
        lifetime_s=SHORT_LIFETIME_S * 4,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        almost_stale = time.time() - locking_module.CLOCK_SLOP_S + SHORT_SLEEP_S / 2
        os.utime(lock_path, (almost_stale, almost_stale))

        with pytest.raises(LockError, match="could not acquire lock"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ):
                pass


def test_stale_break_refuses_malformed_manifest(tmp_path: Path) -> None:
    lock_path = tmp_path / "broken.lock"
    lock_path.write_text("not json", encoding="utf-8")
    stale_time = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (stale_time, stale_time))

    with pytest.raises(LockError, match="malformed lock manifest"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            pass

    assert lock_path.read_text(encoding="utf-8") == "not json"


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
    lock_path = tmp_path / "single.lock"

    with patch("os.link", side_effect=OSError(errno.EINVAL, "bad link")):
        with pytest.raises(OSError, match="bad link"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                pass


def test_release_raises_lock_lost_for_estale_unlink_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"
    original_unlink = os.unlink
    call_count = 0

    def flaky_unlink(path, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError(errno.ESTALE, "stale unlink")
        return original_unlink(path, *args, **kwargs)

    with patch("os.unlink", side_effect=flaky_unlink):
        with pytest.raises(LockError, match="lost lock"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ):
                assert lock_path.exists()


def test_release_raises_unexpected_unlink_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"
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


def test_process_exit_without_cleanup_allows_reclaim_after_expiry(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"
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

        with pytest.raises(LockError, match="could not acquire lock"):
            with lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ):
                pass

        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            acquire_timeout_s=SHORT_LIFETIME_S
            + locking_module.CLOCK_SLOP_S
            + SHORT_SLEEP_S,
            acquire_poll_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            second_owner = str(_manifest_claim_path(lock_path))
            assert second_owner != first_owner
    finally:
        proc.join(timeout=PROCESS_TIMEOUT_S)
