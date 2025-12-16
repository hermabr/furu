import json
import datetime

import huldra
from huldra.storage.state import _StateResultSuccess


class Dummy(huldra.Huldra[int]):
    _create_calls: int = 0
    _load_calls: int = 0

    def _create(self) -> int:
        object.__setattr__(self, "_create_calls", self._create_calls + 1)
        value = 123
        (self.huldra_dir / "value.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        object.__setattr__(self, "_load_calls", self._load_calls + 1)
        return json.loads((self.huldra_dir / "value.json").read_text())


def test_load_or_create_returns_create_result_without_load(huldra_tmp_root) -> None:
    obj = Dummy()
    result = obj.load_or_create()

    assert result == 123
    assert obj._create_calls == 1
    assert obj._load_calls == 0

    result2 = obj.load_or_create()
    assert result2 == 123
    assert obj._create_calls == 1
    assert obj._load_calls == 1


def test_exists_reflects_success_state(huldra_tmp_root) -> None:
    obj = Dummy()
    assert obj.exists() is False
    obj.load_or_create()
    assert obj.exists() is True


def test_load_or_create_recovers_from_expired_running_lease(huldra_tmp_root) -> None:
    obj = Dummy()
    directory = obj.huldra_dir
    directory.mkdir(parents=True, exist_ok=True)

    expired = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=60)
    ).isoformat(timespec="seconds")

    attempt_id = huldra.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=0.05,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    def mutate(state) -> None:
        attempt = state.attempt
        assert attempt is not None
        assert attempt.id == attempt_id
        attempt.started_at = expired
        attempt.heartbeat_at = expired
        attempt.lease_duration_sec = 0.05
        attempt.lease_expires_at = expired

    huldra.StateManager.update_state(directory, mutate)
    lock_path = huldra.StateManager.get_lock_path(
        directory, huldra.StateManager.COMPUTE_LOCK
    )
    lock_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "host": "other-host",
                "created_at": expired,
                "lock_id": "x",
            }
        )
        + "\n"
    )

    result = obj.load_or_create()
    assert result == 123
    assert isinstance(huldra.StateManager.read_state(directory).result, _StateResultSuccess)


def test_load_or_create_waits_until_lease_expires_then_recovers(
    huldra_tmp_root,
) -> None:
    obj = Dummy()
    directory = obj.huldra_dir
    directory.mkdir(parents=True, exist_ok=True)

    # Simulate another process holding the compute lock, but with a lease that will expire.
    lock_path = huldra.StateManager.get_lock_path(
        directory, huldra.StateManager.COMPUTE_LOCK
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "host": "other-host",
                "created_at": "x",
                "lock_id": "x",
            }
        )
        + "\n"
    )
    soon = (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=0.02)
    ).isoformat(timespec="seconds")

    attempt_id = huldra.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=0.02,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    def mutate2(state) -> None:
        attempt = state.attempt
        assert attempt is not None
        assert attempt.id == attempt_id
        attempt.started_at = soon
        attempt.heartbeat_at = soon
        attempt.lease_duration_sec = 0.02
        attempt.lease_expires_at = soon

    huldra.StateManager.update_state(directory, mutate2)

    result = obj.load_or_create()
    assert result == 123
    assert lock_path.exists() is False
    assert isinstance(huldra.StateManager.read_state(directory).result, _StateResultSuccess)
