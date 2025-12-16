import json
import socket

import huldra
import pytest
from huldra.storage.state import _StateResultAbsent, _StateResultIncomplete


def test_state_default_and_attempt_lifecycle(huldra_tmp_root, tmp_path) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()

    state0 = huldra.StateManager.read_state(directory)
    assert state0.schema_version == huldra.StateManager.SCHEMA_VERSION
    assert isinstance(state0.result, _StateResultAbsent)
    assert state0.attempt is None

    attempt_id = huldra.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=0.05,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )
    state1 = huldra.StateManager.read_state(directory)
    assert isinstance(state1.result, _StateResultIncomplete)
    assert state1.attempt is not None
    assert state1.attempt.id == attempt_id
    assert state1.attempt.status == "running"
    assert state1.updated_at is not None


def test_locks_are_exclusive(huldra_tmp_root, tmp_path) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()
    lock_path = huldra.StateManager.get_lock_path(
        directory, huldra.StateManager.COMPUTE_LOCK
    )

    fd1 = huldra.StateManager.try_lock(lock_path)
    assert fd1 is not None
    assert huldra.StateManager.try_lock(lock_path) is None

    huldra.StateManager.release_lock(fd1, lock_path)
    assert lock_path.exists() is False


def test_reconcile_marks_dead_local_attempt_as_crashed(
    huldra_tmp_root, tmp_path
) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()

    attempt_id = huldra.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": socket.gethostname(), "user": "x"},
        scheduler={},
    )

    assert (
        huldra.StateManager.heartbeat(
            directory, attempt_id="wrong", lease_duration_sec=60.0
        )
        is False
    )
    assert (
        huldra.StateManager.heartbeat(
            directory, attempt_id=attempt_id, lease_duration_sec=60.0
        )
        is True
    )

    # If reconcile decides the attempt is dead, it clears the compute lock.
    lock_path = huldra.StateManager.get_lock_path(
        directory, huldra.StateManager.COMPUTE_LOCK
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
    state2 = huldra.StateManager.reconcile(directory)
    assert state2.attempt is not None
    assert state2.attempt.status == "crashed"
    assert lock_path.exists() is False


def test_state_warns_when_retrying_after_failure(
    huldra_tmp_root, tmp_path, capsys
) -> None:
    pytest.importorskip("rich")

    directory = tmp_path / "obj"
    directory.mkdir()

    attempt_id = huldra.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": socket.gethostname(), "user": "x"},
        scheduler={},
    )
    huldra.StateManager.finish_attempt_failed(
        directory,
        attempt_id=attempt_id,
        error={"type": "RuntimeError", "message": "boom"},
    )
    capsys.readouterr()

    huldra.StateManager.start_attempt_running(
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
        in (huldra.HULDRA_CONFIG.base_root / "huldra.log").read_text()
    )


def test_state_warns_when_restart_after_stale_pid(
    huldra_tmp_root, tmp_path, capsys
) -> None:
    pytest.importorskip("rich")

    directory = tmp_path / "obj"
    directory.mkdir()

    huldra.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": socket.gethostname(), "user": "x"},
        scheduler={},
    )
    huldra.StateManager.reconcile(directory)
    capsys.readouterr()

    huldra.StateManager.start_attempt_running(
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
        in (huldra.HULDRA_CONFIG.base_root / "huldra.log").read_text()
    )
