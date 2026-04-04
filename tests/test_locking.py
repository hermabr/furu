import os
import threading
import time
from contextlib import contextmanager
from contextlib import suppress
from multiprocessing import Process, Queue, get_context
from pathlib import Path

import pytest

import furu.locking as locking_module
from furu.locking import (
    Lease,
    LockAcquireError,
    LockLostError,
    NotLockedError,
    lock,
    lock_many,
)
from furu.metadata import LockClaim

TEST_TIMING_SCALE = 4.0 if os.environ.get("GITHUB_ACTIONS") == "true" else 1.0
TEST_CLOCK_SLOP_S = 0.02
SHORT_SLEEP_S = 0.05 * TEST_TIMING_SCALE
SHORT_LIFETIME_S = 0.05 * TEST_TIMING_SCALE
SHORT_HEARTBEAT_INTERVAL_S = 0.02 * TEST_TIMING_SCALE
PROCESS_TIMEOUT_S = 0.5 * TEST_TIMING_SCALE


@pytest.fixture(autouse=True)
def _small_clock_slop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locking_module, "CLOCK_SLOP_S", TEST_CLOCK_SLOP_S)


def _read_claim(path: Path) -> LockClaim:
    return LockClaim.model_validate_json(path.read_text(encoding="utf-8"))


def _claim_path_for(lock_path: Path) -> Path:
    return _read_claim(lock_path).claim_path


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
    lock_path: Path,
    owner_path: str,
    *,
    lifetime_s: float = SHORT_LIFETIME_S,
) -> None:
    with lock(
        lock_path,
        lifetime_s=lifetime_s,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        Path(owner_path).write_text(str(_claim_path_for(lock_path)), encoding="utf-8")
        os._exit(0)


def _child_hold_lock_and_report_heartbeat(
    lock_path: Path,
    thread_queue: Queue,
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
            heartbeat_threads = [
                thread.name
                for thread in threading.enumerate()
                if thread.name.startswith("lock-heartbeat:")
            ]
            assert len(heartbeat_threads) == 1
            thread_queue.put((os.getpid(), heartbeat_threads[0]))
            release_queue.get()


def _child_hold_batch(
    lock_paths: list[Path],
    start_evt,
    ready_queue: Queue,
    release_queue: Queue,
) -> None:
    with suppress(NotLockedError, LockLostError):
        ready_queue.put(("ready", os.getpid()))
        start_evt.wait()
        with lock_many(
            lock_paths,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            ready_queue.put(("held", os.getpid(), [str(path) for path in lock_paths]))
            release_queue.get()


def _drop_current_lock(lock_path: Path) -> None:
    claim_path = _claim_path_for(lock_path)
    lock_path.unlink()
    claim_path.unlink()


def test_lock_many_creates_one_claim_file_per_lock_path(tmp_path: Path) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]

    with lock_many(
        lock_paths,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as lease:
        assert lease.held()

        claim_paths = [_claim_path_for(lock_path) for lock_path in lock_paths]
        assert len(set(claim_paths)) == len(lock_paths)

        for lock_path, claim_path in zip(lock_paths, claim_paths, strict=True):
            claim = _read_claim(claim_path)
            assert claim.lock_path == lock_path.resolve()
            assert claim.claim_path == claim_path
            assert os.path.samestat(lock_path.stat(), claim_path.stat())


def test_lock_many_normalizes_paths_and_keeps_distinct_claims(tmp_path: Path) -> None:
    lock_a = tmp_path / "a.lock"
    lock_b = tmp_path / "b.lock"

    with lock_many([lock_b, lock_a, lock_a]) as lease:
        assert [entry.lock_path for entry in lease.entries] == [
            lock_a.resolve(),
            lock_b.resolve(),
        ]
        assert len({entry.claim_path for entry in lease.entries}) == 2
        assert lease.held()


def test_lock_wrapper_delegates_to_lock_many(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[Path], float | None]] = []
    fake_lease = Lease(())

    @contextmanager
    def fake_lock_many(
        lock_paths: list[Path],
        *,
        acquire_timeout_s: float | None = None,
        **_: object,
    ):
        calls.append((lock_paths, acquire_timeout_s))
        yield fake_lease

    monkeypatch.setattr(locking_module, "lock_many", fake_lock_many)

    with lock(tmp_path / "single.lock", acquire_timeout_s=1.0) as lease:
        assert lease is fake_lease

    assert calls == [([tmp_path / "single.lock"], 1.0)]


def test_lock_uses_default_arguments(tmp_path: Path) -> None:
    with lock(tmp_path / "single.lock") as lease:
        assert lease.held()


def test_lease_held_returns_false_when_lock_is_lost(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with pytest.raises(LockLostError, match="lost lock"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ) as lease:
            _drop_current_lock(lock_path)
            assert not lease.held()


def test_lease_assert_held_raises_lock_lost_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with pytest.raises(LockLostError, match="custom message"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ) as lease:
            _drop_current_lock(lock_path)
            lease.assert_held("custom message")


def test_exit_raises_lock_lost_error_when_lock_is_lost_mid_block(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"

    with pytest.raises(LockLostError, match="lost lock"):
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
    ) as lease:
        assert lease.held()
        time.sleep(SHORT_LIFETIME_S * 4)

        with pytest.raises(LockAcquireError):
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

        with pytest.raises(LockAcquireError):
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
        with pytest.raises(LockAcquireError):
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
        ) as lease:
            assert lease.held()
            assert lock_path.exists()
    finally:
        holder.join(timeout=PROCESS_TIMEOUT_S)


def test_stale_lock_can_be_reclaimed_after_expiry(tmp_path: Path) -> None:
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

        with pytest.raises(LockAcquireError):
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
            second_owner = str(_claim_path_for(lock_path))
            assert second_owner != first_owner
    finally:
        proc.join(timeout=PROCESS_TIMEOUT_S)


def test_stale_break_refuses_malformed_claim_json(tmp_path: Path) -> None:
    lock_path = tmp_path / "broken.lock"
    lock_path.write_text("not json", encoding="utf-8")
    stale_time = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (stale_time, stale_time))

    with pytest.raises(LockAcquireError, match="malformed lock claim"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            pass

    assert lock_path.read_text(encoding="utf-8") == "not json"


def test_cross_directory_lock_requests_are_allowed_even_if_parents_look_cross_fs(
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
        if path == second_parent:
            return _FakeStat(stat_result, st_dev=stat_result.st_dev + 1)
        return stat_result

    monkeypatch.setattr(type(first_parent), "stat", fake_stat)

    with lock_many(lock_paths) as lease:
        assert lease.held()


def test_partial_acquire_rollback_releases_subset_under_overlap(tmp_path: Path) -> None:
    first_lock = tmp_path / "a.lock"
    blocked_lock = tmp_path / "b.lock"

    with lock(
        blocked_lock,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        with pytest.raises(LockAcquireError):
            with lock_many(
                [first_lock, blocked_lock],
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ):
                pass

    assert not first_lock.exists()


def test_overlapping_batches_do_not_deadlock(tmp_path: Path) -> None:
    lock_a = tmp_path / "a.lock"
    lock_b = tmp_path / "b.lock"
    lock_c = tmp_path / "c.lock"
    ctx = get_context("spawn")
    start_evt = ctx.Event()
    ready_queue = ctx.Queue()
    release_first = ctx.Queue()
    release_second = ctx.Queue()
    proc_first = ctx.Process(
        target=_child_hold_batch,
        args=([lock_a, lock_b], start_evt, ready_queue, release_first),
    )
    proc_second = ctx.Process(
        target=_child_hold_batch,
        args=([lock_b, lock_c], start_evt, ready_queue, release_second),
    )
    proc_first.start()
    proc_second.start()

    try:
        ready = [
            ready_queue.get(timeout=PROCESS_TIMEOUT_S),
            ready_queue.get(timeout=PROCESS_TIMEOUT_S),
        ]
        assert all(event[0] == "ready" for event in ready)

        start_evt.set()

        first_held = ready_queue.get(timeout=PROCESS_TIMEOUT_S * 2)
        assert first_held[0] == "held"
        if first_held[2] == [str(lock_a), str(lock_b)]:
            release_first.put(True)
        else:
            release_second.put(True)

        second_held = ready_queue.get(timeout=PROCESS_TIMEOUT_S * 2)
        assert second_held[0] == "held"
        if second_held[2] == [str(lock_a), str(lock_b)]:
            release_first.put(True)
        else:
            release_second.put(True)
    finally:
        proc_first.join(timeout=PROCESS_TIMEOUT_S)
        proc_second.join(timeout=PROCESS_TIMEOUT_S)

    assert proc_first.exitcode == 0
    assert proc_second.exitcode == 0
