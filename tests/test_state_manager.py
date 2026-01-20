import json
import socket
import threading
import time

import furu
import pytest
from furu.errors import FuruLockNotAcquired, FuruWaitTimeout
from furu.storage.state import (
    _StateResultAbsent,
    _StateResultIncomplete,
    _StateResultSuccess,
    compute_lock,
)


def test_state_default_and_attempt_lifecycle(furu_tmp_root, tmp_path) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()

    state0 = furu.StateManager.read_state(directory)
    assert state0.schema_version == furu.StateManager.SCHEMA_VERSION
    assert isinstance(state0.result, _StateResultAbsent)
    assert state0.attempt is None

    attempt_id = furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=0.05,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )
    state1 = furu.StateManager.read_state(directory)
    assert isinstance(state1.result, _StateResultIncomplete)
    assert state1.attempt is not None
    assert state1.attempt.id == attempt_id
    assert state1.attempt.status == "running"
    assert state1.updated_at is not None


def test_locks_are_exclusive(furu_tmp_root, tmp_path) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()
    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )

    fd1 = furu.StateManager.try_lock(lock_path)
    assert fd1 is not None
    assert furu.StateManager.try_lock(lock_path) is None

    furu.StateManager.release_lock(fd1, lock_path)
    assert lock_path.exists() is False


def test_reconcile_marks_dead_local_attempt_as_crashed(furu_tmp_root, tmp_path) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()

    attempt_id = furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": socket.gethostname(), "user": "x"},
        scheduler={},
    )

    assert (
        furu.StateManager.heartbeat(
            directory, attempt_id="wrong", lease_duration_sec=60.0
        )
        is False
    )
    assert (
        furu.StateManager.heartbeat(
            directory, attempt_id=attempt_id, lease_duration_sec=60.0
        )
        is True
    )

    # If reconcile decides the attempt is dead, it clears the compute lock.
    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )
    lock_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "host": socket.gethostname(),
                "created_at": "x",
                "lock_id": "x",
            }
        )
        + "\n"
    )
    state2 = furu.StateManager.reconcile(directory)
    assert state2.attempt is not None
    assert state2.attempt.status == "crashed"
    assert lock_path.exists() is False


def test_state_warns_when_retrying_after_failure(
    furu_tmp_root, tmp_path, capsys
) -> None:
    pytest.importorskip("rich")

    directory = tmp_path / "obj"
    directory.mkdir()

    attempt_id = furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": socket.gethostname(), "user": "x"},
        scheduler={},
    )
    furu.StateManager.finish_attempt_failed(
        directory,
        attempt_id=attempt_id,
        error={"type": "RuntimeError", "message": "boom"},
    )
    capsys.readouterr()

    furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": socket.gethostname(), "user": "x"},
        scheduler={},
    )
    err = capsys.readouterr().err
    assert "state: retrying after previous failure" in err
    assert (
        "state: retrying after previous failure"
        in (furu.FURU_CONFIG.base_root / "furu.log").read_text()
    )


def test_state_warns_when_restart_after_stale_pid(
    furu_tmp_root, tmp_path, capsys
) -> None:
    pytest.importorskip("rich")

    directory = tmp_path / "obj"
    directory.mkdir()

    furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": socket.gethostname(), "user": "x"},
        scheduler={},
    )
    furu.StateManager.reconcile(directory)
    capsys.readouterr()

    furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": socket.gethostname(), "user": "x"},
        scheduler={},
    )
    err = capsys.readouterr().err
    assert "state: restarting after stale attempt (pid_dead)" in err
    assert (
        "state: restarting after stale attempt (pid_dead)"
        in (furu.FURU_CONFIG.base_root / "furu.log").read_text()
    )


# --- compute_lock context manager tests ---


def test_compute_lock_acquires_lock_and_records_attempt(
    furu_tmp_root, tmp_path
) -> None:
    """Test that compute_lock atomically acquires lock and records attempt."""
    directory = tmp_path / "obj"
    directory.mkdir()

    with compute_lock(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        heartbeat_interval_sec=10.0,
        owner={"pid": 12345, "host": "test-host", "user": "test-user"},
    ) as ctx:
        # Lock file should exist
        lock_path = furu.StateManager.get_lock_path(
            directory, furu.StateManager.COMPUTE_LOCK
        )
        assert lock_path.exists()

        # Attempt should be recorded
        state = furu.StateManager.read_state(directory)
        assert state.attempt is not None
        assert state.attempt.id == ctx.attempt_id
        assert state.attempt.status == "running"
        assert isinstance(state.result, _StateResultIncomplete)

    # After exiting, lock should be released
    assert not lock_path.exists()


def test_compute_lock_releases_on_exception(furu_tmp_root, tmp_path) -> None:
    """Test that compute_lock releases lock even when exception is raised."""
    directory = tmp_path / "obj"
    directory.mkdir()

    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )

    with pytest.raises(ValueError, match="test error"):
        with compute_lock(
            directory,
            backend="local",
            lease_duration_sec=60.0,
            heartbeat_interval_sec=10.0,
            owner={"pid": 12345, "host": "test-host", "user": "test-user"},
        ):
            raise ValueError("test error")

    # Lock should be released even after exception
    assert not lock_path.exists()


def test_compute_lock_cleans_orphaned_lock(furu_tmp_root, tmp_path) -> None:
    """Test that compute_lock cleans up orphaned lock files (lock exists but no active attempt)."""
    directory = tmp_path / "obj"
    directory.mkdir()

    # Create an orphaned lock file (no corresponding attempt in state)
    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "host": "old-host",
                "created_at": "2020-01-01T00:00:00+00:00",
                "lock_id": "orphaned",
            }
        )
        + "\n"
    )

    # compute_lock should detect orphaned lock and clean it up
    with compute_lock(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        heartbeat_interval_sec=10.0,
        owner={"pid": 12345, "host": "test-host", "user": "test-user"},
    ) as ctx:
        # Should have acquired lock successfully
        assert ctx.attempt_id is not None
        state = furu.StateManager.read_state(directory)
        assert state.attempt is not None
        assert state.attempt.status == "running"


def test_compute_lock_raises_on_success_state(furu_tmp_root, tmp_path) -> None:
    """Test that compute_lock raises FuruLockNotAcquired if experiment already succeeded."""
    directory = tmp_path / "obj"
    directory.mkdir()

    # Create a successful state
    attempt_id = furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 12345, "host": "test-host", "user": "test-user"},
    )
    furu.StateManager.write_success_marker(directory, attempt_id=attempt_id)
    furu.StateManager.finish_attempt_success(directory, attempt_id=attempt_id)

    # Verify state is success
    state = furu.StateManager.read_state(directory)
    assert isinstance(state.result, _StateResultSuccess)

    # Create a lock file to simulate contention
    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": 99999, "host": "x", "created_at": "x", "lock_id": "x"})
        + "\n"
    )

    with pytest.raises(FuruLockNotAcquired, match="already succeeded"):
        with compute_lock(
            directory,
            backend="local",
            lease_duration_sec=60.0,
            heartbeat_interval_sec=10.0,
            owner={"pid": 12345, "host": "test-host", "user": "test-user"},
        ):
            pass


def test_compute_lock_timeout(furu_tmp_root, tmp_path) -> None:
    """Test that compute_lock raises FuruWaitTimeout when max_wait_time_sec is exceeded."""
    directory = tmp_path / "obj"
    directory.mkdir()

    # Start an attempt that holds the lock (simulating another process)
    furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,  # Long lease so it won't expire
        owner={"pid": 99999, "host": "other-host", "user": "other-user"},
    )

    # Acquire the actual lock file
    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )
    lock_fd = furu.StateManager.try_lock(lock_path)
    assert lock_fd is not None

    try:
        # Try to acquire with a very short timeout
        with pytest.raises(FuruWaitTimeout, match="Timed out"):
            with compute_lock(
                directory,
                backend="local",
                lease_duration_sec=60.0,
                heartbeat_interval_sec=10.0,
                owner={"pid": 12345, "host": "test-host", "user": "test-user"},
                max_wait_time_sec=0.1,
                poll_interval_sec=0.05,
            ):
                pass
    finally:
        furu.StateManager.release_lock(lock_fd, lock_path)


def test_compute_lock_is_exclusive(furu_tmp_root, tmp_path) -> None:
    """Test that two concurrent compute_lock calls are mutually exclusive."""
    directory = tmp_path / "obj"
    directory.mkdir()

    results: list[str] = []
    errors: list[Exception] = []

    def worker(worker_id: str, delay_before_release: float) -> None:
        try:
            with compute_lock(
                directory,
                backend="local",
                lease_duration_sec=60.0,
                heartbeat_interval_sec=10.0,
                owner={"pid": 12345, "host": f"host-{worker_id}", "user": "test-user"},
                max_wait_time_sec=5.0,
                poll_interval_sec=0.01,
                reconcile_fn=lambda d: (furu.StateManager.reconcile(d), None)[1],
            ):
                results.append(f"{worker_id}-acquired")
                time.sleep(delay_before_release)
                results.append(f"{worker_id}-released")
        except Exception as e:
            errors.append(e)

    # Start first worker
    t1 = threading.Thread(target=worker, args=("A", 0.2))
    t1.start()

    # Give first worker time to acquire lock
    time.sleep(0.05)

    # Start second worker - should block until first releases
    t2 = threading.Thread(target=worker, args=("B", 0.1))
    t2.start()

    t1.join()
    t2.join()

    assert not errors, f"Unexpected errors: {errors}"

    # Verify mutual exclusion: A should fully complete before B starts
    assert results == ["A-acquired", "A-released", "B-acquired", "B-released"], (
        f"Expected sequential execution but got: {results}"
    )


def test_compute_lock_waits_for_active_attempt(furu_tmp_root, tmp_path) -> None:
    """Test that compute_lock waits for an active attempt and lock to be released."""
    directory = tmp_path / "obj"
    directory.mkdir()

    # Start an attempt with a long lease on a different host (so reconcile can't detect it as dead)
    furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,  # Long lease
        owner={"pid": 99999, "host": "other-host", "user": "other-user"},
    )

    # Acquire the lock (simulating the other process)
    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )
    lock_fd = furu.StateManager.try_lock(lock_path)
    assert lock_fd is not None

    # Release lock in a background thread after a short delay
    def release_later():
        time.sleep(0.1)
        furu.StateManager.release_lock(lock_fd, lock_path)

    release_thread = threading.Thread(target=release_later)
    release_thread.start()

    # compute_lock should wait and eventually acquire
    start = time.time()
    with compute_lock(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        heartbeat_interval_sec=10.0,
        owner={"pid": 12345, "host": "test-host", "user": "test-user"},
        max_wait_time_sec=5.0,
        poll_interval_sec=0.02,
        reconcile_fn=lambda d: (furu.StateManager.reconcile(d), None)[1],
    ) as ctx:
        elapsed = time.time() - start
        # Should have waited for the lock (at least 0.05s, probably ~0.1s)
        assert elapsed >= 0.05
        assert ctx.attempt_id is not None

    release_thread.join()


def test_compute_lock_heartbeat_runs(furu_tmp_root, tmp_path) -> None:
    """Test that the heartbeat thread updates the lease while lock is held."""
    directory = tmp_path / "obj"
    directory.mkdir()

    with compute_lock(
        directory,
        backend="local",
        lease_duration_sec=0.5,
        heartbeat_interval_sec=0.05,  # Heartbeat every 50ms
        owner={"pid": 12345, "host": "test-host", "user": "test-user"},
    ):
        state_before = furu.StateManager.read_state(directory)
        assert state_before.attempt is not None
        assert state_before.attempt.status == "running"
        initial_lease_expires = state_before.attempt.lease_expires_at  # type: ignore[union-attr]

        # Wait long enough for heartbeat to update (need >1 second for ISO timestamp change)
        time.sleep(1.1)

        state_after = furu.StateManager.read_state(directory)
        assert state_after.attempt is not None
        updated_lease_expires = state_after.attempt.lease_expires_at  # type: ignore[union-attr]

        # Lease expiry should have been extended by heartbeat
        assert updated_lease_expires != initial_lease_expires
