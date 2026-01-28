import json
import time

import furu
from furu.adapters import SubmititAdapter


class FakeJob:
    def __init__(self):
        self.job_id: str | None = None

    def done(self) -> bool:
        return False


class DummyTask(furu.Furu[int]):
    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


def test_job_id_watcher_updates_new_attempt(furu_tmp_root) -> None:
    obj = DummyTask()
    directory = obj._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)

    adapter = SubmititAdapter(executor=None)
    job = FakeJob()

    queued_id = furu.StateManager.start_attempt_queued(
        directory,
        backend="submitit",
        lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    adapter.watch_job_id(job, directory, attempt_id=queued_id)

    running_id = furu.StateManager.start_attempt_running(
        directory,
        backend="submitit",
        lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    job.job_id = "job-123"
    deadline = time.time() + 2.0
    while time.time() < deadline:
        state = furu.StateManager.read_state(directory)
        attempt = state.attempt
        if attempt is not None and attempt.scheduler.get("job_id") == "job-123":
            break
        time.sleep(0.1)

    state = furu.StateManager.read_state(directory)
    attempt = state.attempt
    assert attempt is not None
    assert attempt.id == running_id
    assert attempt.scheduler.get("job_id") == "job-123"
