import json

import pytest

import furu
from furu.adapters import SubmititAdapter
from furu.execution.slurm_dag import (
    _job_id_for_in_progress,
    _wait_for_job_id,
    submit_slurm_dag,
)
from furu.execution.slurm_spec import SlurmSpec, SlurmSpecExtraValue
from furu.execution.submitit_factory import make_executor_for_spec
from furu.storage.state import FuruErrorState, _StateAttemptFailed


class FakeExecutor:
    def __init__(self, folder: str):
        self.folder = folder
        self.parameters: dict[str, SlurmSpecExtraValue] = {}

    def update_parameters(self, **kwargs: SlurmSpecExtraValue) -> None:
        self.parameters.update(kwargs)


class FakeJob:
    def __init__(self, job_id: str):
        self.job_id = job_id


class DagTask(furu.Furu[int]):
    name: str = furu.chz.field()
    deps: list[furu.Furu[int]] = furu.chz.field(default_factory=list)

    def _dependencies(self) -> furu.DependencySpec:
        return self.deps

    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(self.name))
        return 1

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


class GpuTask(DagTask):
    def _executor_spec_key(self) -> str:
        return "gpu"


def test_make_executor_for_spec_sets_parameters(tmp_path, monkeypatch) -> None:
    import submitit

    monkeypatch.setattr(submitit, "AutoExecutor", FakeExecutor)
    spec = SlurmSpec(
        partition="cpu",
        gpus=1,
        cpus=8,
        mem_gb=32,
        time_min=90,
        extra={"qos": "high"},
    )

    executor = make_executor_for_spec(
        "default",
        spec,
        kind="nodes",
        submitit_root=tmp_path,
    )

    assert executor.folder == str(tmp_path / "nodes" / "default")
    assert executor.parameters["slurm_partition"] == "cpu"
    assert executor.parameters["cpus_per_task"] == 8
    assert executor.parameters["mem_gb"] == 32
    assert executor.parameters["timeout_min"] == 90
    assert executor.parameters["gpus_per_node"] == 1
    assert executor.parameters["qos"] == "high"


def test_make_executor_for_spec_allows_nested_extra(tmp_path, monkeypatch) -> None:
    import submitit

    monkeypatch.setattr(submitit, "AutoExecutor", FakeExecutor)
    spec = SlurmSpec(
        partition="cpu",
        extra={"slurm_additional_parameters": {"qos": "urgent"}},
    )

    executor = make_executor_for_spec(
        "default",
        spec,
        kind="nodes",
        submitit_root=tmp_path,
    )

    assert executor.parameters["slurm_additional_parameters"] == {"qos": "urgent"}


def test_submit_slurm_dag_wires_dependencies(furu_tmp_root, monkeypatch) -> None:
    leaf = DagTask(name="leaf")
    root = DagTask(name="root", deps=[leaf])

    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    executors: list[FakeExecutor] = []

    def fake_make_executor(
        spec_key: str,
        spec: SlurmSpec,
        *,
        kind: str,
        submitit_root,
        run_id: str | None = None,
    ):
        executor = FakeExecutor(folder=f"{kind}:{spec_key}")
        executors.append(executor)
        return executor

    job_ids = {"leaf": "job-leaf", "root": "job-root"}

    def fake_submit_once(self, adapter, directory, on_job_id, *, allow_failed):
        furu.StateManager.ensure_internal_dir(directory)
        furu.StateManager.start_attempt_queued(
            directory,
            backend="submitit",
            lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
            owner={"pid": 99999, "host": "other-host", "user": "x"},
            scheduler={"job_id": job_ids[self.name]},
        )
        return FakeJob(job_ids[self.name])

    monkeypatch.setattr(
        "furu.execution.slurm_dag.make_executor_for_spec", fake_make_executor
    )
    monkeypatch.setattr(DagTask, "_submit_once", fake_submit_once)

    submission = submit_slurm_dag([root], specs=specs, submitit_root=None)

    assert submission.job_id_by_hash[leaf.furu_hash] == "job-leaf"
    assert submission.job_id_by_hash[root.furu_hash] == "job-root"
    assert submission.root_job_ids[root.furu_hash] == "job-root"
    assert executors[1].parameters["slurm_additional_parameters"] == {
        "dependency": "afterok:job-leaf"
    }


def test_submit_slurm_dag_uses_in_progress_job_ids(furu_tmp_root, monkeypatch) -> None:
    dep = DagTask(name="dep")
    root = DagTask(name="root", deps=[dep])

    directory = dep._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)
    furu.StateManager.start_attempt_queued(
        directory,
        backend="submitit",
        lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={"job_id": "job-dep"},
    )

    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    executors: list[FakeExecutor] = []

    def fake_make_executor(
        spec_key: str,
        spec: SlurmSpec,
        *,
        kind: str,
        submitit_root,
        run_id: str | None = None,
    ):
        executor = FakeExecutor(folder=f"{kind}:{spec_key}")
        executors.append(executor)
        return executor

    def fake_submit_once(self, adapter, directory, on_job_id, *, allow_failed):
        furu.StateManager.ensure_internal_dir(directory)
        furu.StateManager.start_attempt_queued(
            directory,
            backend="submitit",
            lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
            owner={"pid": 99999, "host": "other-host", "user": "x"},
            scheduler={"job_id": "job-root"},
        )
        return FakeJob("job-root")

    monkeypatch.setattr(
        "furu.execution.slurm_dag.make_executor_for_spec", fake_make_executor
    )
    monkeypatch.setattr(DagTask, "_submit_once", fake_submit_once)

    submission = submit_slurm_dag([root], specs=specs, submitit_root=None)

    assert submission.job_id_by_hash[root.furu_hash] == "job-root"
    assert executors[0].parameters["slurm_additional_parameters"] == {
        "dependency": "afterok:job-dep"
    }


def test_submit_slurm_dag_requires_default_spec(furu_tmp_root) -> None:
    with pytest.raises(KeyError):
        submit_slurm_dag([DagTask(name="root")], specs={}, submitit_root=None)


def test_submit_slurm_dag_missing_spec_key_raises(furu_tmp_root, monkeypatch) -> None:
    root = GpuTask(name="root")
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

    monkeypatch.setattr(
        "furu.execution.slurm_dag.make_executor_for_spec",
        lambda *args, **kwargs: FakeExecutor("x"),
    )
    monkeypatch.setattr(GpuTask, "_submit_once", lambda *args, **kwargs: FakeJob("job"))

    with pytest.raises(KeyError):
        submit_slurm_dag([root], specs=specs, submitit_root=None)


def test_submit_slurm_dag_in_progress_requires_submitit_backend(
    furu_tmp_root, monkeypatch
) -> None:
    dep = DagTask(name="dep")
    root = DagTask(name="root", deps=[dep])
    directory = dep._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)

    furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

    monkeypatch.setattr(
        "furu.execution.slurm_dag.make_executor_for_spec",
        lambda *args, **kwargs: FakeExecutor("x"),
    )
    monkeypatch.setattr(DagTask, "_submit_once", lambda *args, **kwargs: FakeJob("x"))

    with pytest.raises(furu.FuruExecutionError, match="non-submitit"):
        submit_slurm_dag([root], specs=specs, submitit_root=None)


def test_submit_slurm_dag_merges_additional_parameters(
    furu_tmp_root, monkeypatch
) -> None:
    leaf = DagTask(name="leaf")
    root = DagTask(name="root", deps=[leaf])

    specs = {
        "default": SlurmSpec(
            partition="cpu",
            cpus=2,
            mem_gb=4,
            time_min=10,
            extra={"slurm_additional_parameters": {"qos": "high"}},
        )
    }
    executors: list[FakeExecutor] = []

    def fake_make_executor(
        spec_key: str,
        spec: SlurmSpec,
        *,
        kind: str,
        submitit_root,
        run_id: str | None = None,
    ):
        executor = FakeExecutor(folder=f"{kind}:{spec_key}")
        executors.append(executor)
        return executor

    job_ids = {"leaf": "job-leaf", "root": "job-root"}

    def fake_submit_once(self, adapter, directory, on_job_id, *, allow_failed):
        furu.StateManager.ensure_internal_dir(directory)
        furu.StateManager.start_attempt_queued(
            directory,
            backend="submitit",
            lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
            owner={"pid": 99999, "host": "other-host", "user": "x"},
            scheduler={"job_id": job_ids[self.name]},
        )
        return FakeJob(job_ids[self.name])

    monkeypatch.setattr(
        "furu.execution.slurm_dag.make_executor_for_spec", fake_make_executor
    )
    monkeypatch.setattr(DagTask, "_submit_once", fake_submit_once)

    submit_slurm_dag([root], specs=specs, submitit_root=None)

    assert executors[1].parameters["slurm_additional_parameters"] == {
        "qos": "high",
        "dependency": "afterok:job-leaf",
    }


def test_wait_for_job_id_updates_after_attempt_switch(furu_tmp_root) -> None:
    obj = DagTask(name="root")
    directory = obj._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)

    furu.StateManager.start_attempt_queued(
        directory,
        backend="submitit",
        lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    running_ids: list[str] = []

    class SwitchingAdapter(SubmititAdapter):
        def __init__(self) -> None:
            super().__init__(executor=None)

        def load_job(self, directory):
            return FakeJob("job-123")

        def get_job_id(self, job):
            if not running_ids:
                running_ids.append(
                    furu.StateManager.start_attempt_running(
                        directory,
                        backend="submitit",
                        lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
                        owner={"pid": 99999, "host": "other-host", "user": "x"},
                        scheduler={"job_id": "job-old"},
                    )
                )
            return "job-123"

    job_id = _wait_for_job_id(
        obj,
        SwitchingAdapter(),
        None,
        timeout_sec=2.0,
        poll_interval_sec=0.01,
    )

    state = furu.StateManager.read_state(directory)
    attempt = state.attempt

    assert attempt is not None
    assert job_id == "job-123"
    assert attempt.id == running_ids[0]
    assert attempt.scheduler.get("job_id") == "job-123"


def test_wait_for_job_id_returns_on_terminal_attempt(furu_tmp_root) -> None:
    obj = DagTask(name="root")
    directory = obj._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)

    furu.StateManager.start_attempt_queued(
        directory,
        backend="submitit",
        lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    def mark_failed(state) -> None:
        attempt = state.attempt
        if attempt is None:
            raise AssertionError("missing attempt")
        state.attempt = _StateAttemptFailed(
            id=attempt.id,
            number=attempt.number,
            backend=attempt.backend,
            started_at=attempt.started_at,
            lease_duration_sec=attempt.lease_duration_sec,
            lease_expires_at=attempt.lease_expires_at,
            owner=attempt.owner,
            scheduler=attempt.scheduler,
            ended_at=furu.StateManager._iso_now(),
            error=FuruErrorState(type="FuruComputeError", message="boom"),
            reason="boom",
        )

    furu.StateManager.update_state(directory, mark_failed)

    class TerminalAdapter(SubmititAdapter):
        def __init__(self) -> None:
            super().__init__(executor=None)

        def load_job(self, directory):
            return FakeJob("job-999")

    job_id = _wait_for_job_id(
        obj,
        TerminalAdapter(),
        None,
        timeout_sec=1.0,
        poll_interval_sec=0.01,
    )

    assert job_id == "job-999"


def test_job_id_for_in_progress_fails_fast_on_terminal_failed_dependency(
    furu_tmp_root, monkeypatch
) -> None:
    obj = DagTask(name="dep")
    directory = obj._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)

    furu.StateManager.start_attempt_queued(
        directory,
        backend="submitit",
        lease_duration_sec=furu.FURU_CONFIG.lease_duration_sec,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    # Make `_wait_for_job_id` observe a job handle and job_id, but flip the attempt
    # to terminal failed before wiring completes.
    monkeypatch.setattr(SubmititAdapter, "load_job", lambda self, d: FakeJob("job-777"))

    marked: list[bool] = []

    def fake_get_job_id(self, job):
        if not marked:
            marked.append(True)

            def mark_failed(state) -> None:
                attempt = state.attempt
                if attempt is None:
                    raise AssertionError("missing attempt")
                state.attempt = _StateAttemptFailed(
                    id=attempt.id,
                    number=attempt.number,
                    backend=attempt.backend,
                    started_at=attempt.started_at,
                    lease_duration_sec=attempt.lease_duration_sec,
                    lease_expires_at=attempt.lease_expires_at,
                    owner=attempt.owner,
                    scheduler=attempt.scheduler,
                    ended_at=furu.StateManager._iso_now(),
                    error=FuruErrorState(type="FuruComputeError", message="boom"),
                    reason="boom",
                )

            furu.StateManager.update_state(directory, mark_failed)
        return "job-777"

    monkeypatch.setattr(SubmititAdapter, "get_job_id", fake_get_job_id)

    with pytest.raises(furu.FuruExecutionError, match="became terminal"):
        _job_id_for_in_progress(obj)
