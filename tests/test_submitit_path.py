import json
from pathlib import Path
from typing import cast

import huldra
import submitit


class _FakeJob:
    def __init__(self, fn, job_id: str):
        self._fn = fn
        self.job_id = job_id
        self._done = False

    def done(self):
        return self._done

    def state(self):
        return "COMPLETED" if self._done else "RUNNING"

    def wait(self):
        if not self._done:
            self._fn()
            self._done = True

    def result(self, timeout=None):
        self.wait()
        return None


class _FakeExecutor:
    def __init__(self):
        self.folder: Path | None = None
        self._i = 0

    def submit(self, fn):
        self._i += 1
        job = _FakeJob(fn, job_id=f"fake-{self._i}")
        job.wait()
        return job


class Dummy(huldra.Huldra[int], slug="test-submitit-path"):
    value: int = huldra.chz.field(default=7)

    def _create(self) -> int:
        (self.huldra_dir / "value.json").write_text(json.dumps(self.value))
        return self.value

    def _load(self) -> int:
        return json.loads((self.huldra_dir / "value.json").read_text())


def test_load_or_create_with_executor_submits_job(huldra_tmp_root) -> None:
    obj = Dummy(value=11)
    job = obj.load_or_create(executor=cast(submitit.Executor, _FakeExecutor()))

    assert obj.exists() is True
    assert (obj.huldra_dir / huldra.SubmititAdapter.JOB_PICKLE).exists() is True
    assert job is not None
    assert obj.load_or_create() == 11


def test_classify_scheduler_state_cancelled(huldra_tmp_root, monkeypatch) -> None:
    adapter = huldra.SubmititAdapter(executor=None)
    monkeypatch.setattr(huldra.HULDRA_CONFIG, "cancelled_is_preempted", True)
    assert adapter.classify_scheduler_state("CANCELLED") == "preempted"
    monkeypatch.setattr(huldra.HULDRA_CONFIG, "cancelled_is_preempted", False)
    assert adapter.classify_scheduler_state("CANCELLED") == "failed"
