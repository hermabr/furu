import json
import datetime
import os
import time

import furu
from furu.execution.slurm_spec import SlurmSpec, slurm_spec_key
from furu.storage.state import _StateResultSuccess


class Dummy(furu.Furu[int]):
    _create_calls: int = 0
    _load_calls: int = 0

    def _create(self) -> int:
        object.__setattr__(self, "_create_calls", self._create_calls + 1)
        value = 123
        (self.furu_dir / "value.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        object.__setattr__(self, "_load_calls", self._load_calls + 1)
        return json.loads((self.furu_dir / "value.json").read_text())


def test_furu_hash_is_stable(furu_tmp_root) -> None:
    obj = Dummy()
    first = obj.furu_hash
    second = obj.furu_hash

    assert first == second


def test_get_returns_create_result_without_load(furu_tmp_root) -> None:
    obj = Dummy()
    result = obj.get()

    assert result == 123
    assert obj._create_calls == 1
    assert obj._load_calls == 0

    result2 = obj.get()
    assert result2 == 123
    assert obj._create_calls == 1
    assert obj._load_calls == 1


def test_always_rerun_rebuilds_cached_results(furu_tmp_root, monkeypatch) -> None:
    obj = Dummy()
    assert obj.get() == 123
    assert obj._create_calls == 1
    assert obj._load_calls == 0

    qualname = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun", {qualname})

    assert obj.get() == 123
    assert obj._create_calls == 2
    assert obj._load_calls == 0


def test_always_rerun_all_rebuilds_cached_results(furu_tmp_root, monkeypatch) -> None:
    obj = Dummy()
    assert obj.get() == 123
    assert obj._create_calls == 1
    assert obj._load_calls == 0

    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun_all", True)

    assert obj.get() == 123
    assert obj._create_calls == 2
    assert obj._load_calls == 0


def test_exists_reflects_success_state(furu_tmp_root) -> None:
    obj = Dummy()
    assert obj.exists() is False
    obj.get()
    assert obj.exists() is True


def test_get_recovers_from_expired_running_lease(furu_tmp_root) -> None:
    obj = Dummy()
    directory = obj._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)

    expired = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=3600)
    ).isoformat(timespec="seconds")

    attempt_id = furu.StateManager.start_attempt_running(
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
        attempt.lease_duration_sec = 0.05
        attempt.lease_expires_at = expired

    furu.StateManager.update_state(directory, mutate)
    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
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
    expired_ts = datetime.datetime.fromisoformat(expired).timestamp()
    os.utime(lock_path, (expired_ts, expired_ts))

    result = obj.get()
    assert result == 123
    assert isinstance(
        furu.StateManager.read_state(directory).result, _StateResultSuccess
    )


def test_get_waits_until_lease_expires_then_recovers(
    furu_tmp_root,
) -> None:
    obj = Dummy()
    directory = obj._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)

    # Simulate another process holding the compute lock, but with a lease that will expire.
    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )
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
    furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=0.02,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )
    stale_time = time.time() - 3600.0
    os.utime(lock_path, (stale_time, stale_time))

    result = obj.get()
    assert result == 123
    assert lock_path.exists() is False
    assert isinstance(
        furu.StateManager.read_state(directory).result, _StateResultSuccess
    )


def test_submit_once_records_slurm_spec_metadata(furu_tmp_root) -> None:
    class StubJob:
        job_id = "job-stub"

        def done(self) -> bool:
            return True

    class StubExecutor:
        def submit(self, fn):
            _ = fn
            return StubJob()

    obj = Dummy()
    directory = obj._base_furu_dir()
    adapter = furu.SubmititAdapter(executor=StubExecutor())

    obj._submit_once(
        adapter,
        directory=directory,
        on_job_id=None,
        allow_failed=furu.FURU_CONFIG.retry_failed,
    )

    state = furu.StateManager.read_state(directory)
    attempt = state.attempt
    assert attempt is not None
    assert attempt.status == "queued"
    assert attempt.scheduler.get("slurm_spec_key") == slurm_spec_key(SlurmSpec())
    spec = attempt.scheduler.get("slurm_spec")
    assert isinstance(spec, dict)
    assert spec.get("partition") is None
    assert spec.get("gpus") == 0
    assert spec.get("cpus") == 4
    assert spec.get("mem_gb") == 16
    assert spec.get("time_min") == 60
    assert spec.get("extra") == {}
