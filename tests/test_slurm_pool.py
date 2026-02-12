import json
import os
import time
from pathlib import Path
from typing import ClassVar

import pytest

import furu
from furu.execution.plan import DependencyPlan, PlanNode
from furu.execution.slurm_pool import (
    _claim_task,
    _ensure_queue_layout,
    _handle_failed_tasks,
    _mark_done,
    _mark_failed,
    _requeue_stale_running,
    _scan_failed_tasks,
    _spec_with_pool_worker_logs,
    pool_worker_main,
    run_slurm_pool,
)
from furu.execution.slurm_spec import SlurmSpec, slurm_spec_key

DEFAULT_SPEC = SlurmSpec()
DEFAULT_SPEC_KEY = slurm_spec_key(DEFAULT_SPEC)
GPU_SPEC = SlurmSpec(partition="gpu", gpus=1, cpus=8, mem_gb=64, time_min=720)
GPU_SPEC_KEY = slurm_spec_key(GPU_SPEC)


class InlineJob:
    def __init__(self, fn):
        self._done = False
        self.job_id = "inline"
        fn()
        self._done = True

    def done(self) -> bool:
        return self._done


class InlineExecutor:
    def __init__(self):
        self.submitted: int = 0

    def submit(self, fn):
        self.submitted += 1
        return InlineJob(fn)


class NoopJob:
    job_id = "noop"

    def done(self) -> bool:
        return True


class NoopExecutor:
    def submit(self, fn):
        return NoopJob()


class CountingJob:
    job_id = "counting"

    def __init__(self) -> None:
        self.done_calls = 0

    def done(self) -> bool:
        self.done_calls += 1
        return False


class CountingExecutor:
    def __init__(self, job: CountingJob) -> None:
        self._job = job
        self.submitted = 0

    def submit(self, fn):
        self.submitted += 1
        return self._job


class PoolTask(furu.Furu[int]):
    value: int = furu.chz.field()

    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(self.value))
        return self.value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


class FlakyPoolTask(furu.Furu[int]):
    _create_calls: ClassVar[int] = 0

    def _create(self) -> int:
        type(self)._create_calls += 1
        if type(self)._create_calls == 1:
            raise RuntimeError("boom")
        value = 7
        (self.furu_dir / "value.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


class GpuPoolTask(PoolTask):
    def _executor(self) -> SlurmSpec:
        return GPU_SPEC


class QosPoolTask(PoolTask):
    def _executor(self) -> SlurmSpec:
        return SlurmSpec(
            partition="cpu",
            cpus=2,
            mem_gb=4,
            time_min=10,
            extra={"slurm_additional_parameters": {"qos": "high"}},
        )


# def _require_submitit_local_runtime() -> None:
#     pytest.importorskip("submitit")
#     pytest.importorskip("pkg_resources")


def test_spec_with_pool_worker_logs_merges_additional_parameters(tmp_path) -> None:
    base_spec = SlurmSpec(
        partition="cpu",
        extra={"slurm_additional_parameters": {"qos": "high"}},
    )

    worker_dir = tmp_path / "queue" / "running" / DEFAULT_SPEC_KEY / "worker-1"
    worker_spec = _spec_with_pool_worker_logs(base_spec, worker_dir)

    assert worker_spec.extra is not None
    additional = worker_spec.extra.get("slurm_additional_parameters")
    assert isinstance(additional, dict)
    assert additional["qos"] == "high"
    assert additional["output"] == str(worker_dir / "stdout.log")
    assert additional["error"] == str(worker_dir / "stderr.log")


def test_spec_with_pool_worker_logs_requires_mapping_additional_parameters(
    tmp_path,
) -> None:
    bad_spec = SlurmSpec(
        partition="cpu",
        extra={"slurm_additional_parameters": "bad"},
    )

    with pytest.raises(TypeError, match="mapping"):
        _spec_with_pool_worker_logs(
            bad_spec,
            tmp_path / "queue" / "running" / DEFAULT_SPEC_KEY / "worker-1",
        )


def test_run_slurm_pool_executes_tasks(furu_tmp_root, tmp_path, monkeypatch) -> None:
    root = PoolTask(value=3)

    def fake_make_executor(
        spec_key: str,
        spec: SlurmSpec,
        *,
        kind: str,
        submitit_root,
        run_id: str | None = None,
    ):
        return InlineExecutor()

    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        fake_make_executor,
    )

    run = run_slurm_pool(
        [root],
        max_workers_total=1,
        window_size="dfs",
        idle_timeout_sec=0.01,
        poll_interval_sec=0.01,
        submitit_root=None,
        run_root=tmp_path,
    )

    assert root.exists()
    assert (run.run_dir / "queue" / "done" / f"{root.furu_hash}.json").exists()


def test_run_slurm_pool_executes_with_submitit_local_backend(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    # _require_submitit_local_runtime()

    import submitit

    root = PoolTask(value=7)
    original_auto_executor = submitit.AutoExecutor

    class LocalAutoExecutor:
        def __new__(cls, folder: str):
            return original_auto_executor(folder=folder, cluster="local")

    monkeypatch.setattr(submitit, "AutoExecutor", LocalAutoExecutor)

    run = run_slurm_pool(
        [root],
        max_workers_total=1,
        window_size="dfs",
        idle_timeout_sec=0.05,
        poll_interval_sec=0.01,
        submitit_root=None,
        run_root=tmp_path,
    )

    assert root.exists()
    assert (run.run_dir / "queue" / "done" / f"{root.furu_hash}.json").exists()


def test_run_slurm_pool_routes_worker_logs_to_worker_dir(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    root = QosPoolTask(value=4)
    seen_specs: list[SlurmSpec] = []
    seen_spec_keys: list[str] = []

    def fake_make_executor(
        spec_key: str,
        spec: SlurmSpec,
        *,
        kind: str,
        submitit_root,
        run_id: str | None = None,
    ):
        seen_spec_keys.append(spec_key)
        seen_specs.append(spec)
        return InlineExecutor()

    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        fake_make_executor,
    )

    run = run_slurm_pool(
        [root],
        max_workers_total=1,
        window_size="dfs",
        idle_timeout_sec=0.01,
        poll_interval_sec=0.01,
        submitit_root=None,
        run_root=tmp_path,
    )

    assert root.exists()
    assert seen_specs
    assert seen_spec_keys
    worker_spec = seen_specs[0]
    assert worker_spec.extra is not None
    additional = worker_spec.extra.get("slurm_additional_parameters")
    assert isinstance(additional, dict)
    assert additional["qos"] == "high"

    output = additional.get("output")
    error = additional.get("error")
    assert isinstance(output, str)
    assert isinstance(error, str)

    output_path = Path(output)
    error_path = Path(error)
    assert output_path.name == "stdout.log"
    assert error_path.name == "stderr.log"
    assert output_path.parent == error_path.parent
    assert (
        output_path.parent.parent
        == run.run_dir / "queue" / "running" / seen_spec_keys[0]
    )


def test_run_slurm_pool_retries_failed_when_enabled(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    FlakyPoolTask._create_calls = 0
    root = FlakyPoolTask()

    with pytest.raises(RuntimeError, match="boom"):
        root.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)

    def fake_make_executor(
        spec_key: str,
        spec: SlurmSpec,
        *,
        kind: str,
        submitit_root,
        run_id: str | None = None,
    ):
        return InlineExecutor()

    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        fake_make_executor,
    )

    run_slurm_pool(
        [root],
        max_workers_total=1,
        window_size="dfs",
        idle_timeout_sec=0.01,
        poll_interval_sec=0.01,
        submitit_root=None,
        run_root=tmp_path,
    )

    assert root.exists()
    assert FlakyPoolTask._create_calls == 2


def test_run_slurm_pool_fails_fast_on_failed_state_when_retry_disabled(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    FlakyPoolTask._create_calls = 0
    root = FlakyPoolTask()

    with pytest.raises(RuntimeError, match="boom"):
        root.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)
    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        lambda *args, **kwargs: NoopExecutor(),
    )

    with pytest.raises(RuntimeError, match="failed dependencies"):
        run_slurm_pool(
            [root],
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            submitit_root=None,
            run_root=tmp_path,
        )


def test_pool_worker_detects_spec_mismatch(tmp_path) -> None:
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    task = GpuPoolTask(value=1)
    payload = {
        "hash": task.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": task.to_dict(),
    }
    todo_path = (
        tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / f"{task.furu_hash}.json"
    )
    todo_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(RuntimeError):
        pool_worker_main(
            tmp_path,
            DEFAULT_SPEC_KEY,
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
        )

    failed_path = tmp_path / "queue" / "failed" / f"{task.furu_hash}.json"
    assert failed_path.exists()
    payload = json.loads(failed_path.read_text())
    assert payload["failure_kind"] == "protocol"


def test_pool_worker_marks_invalid_json_payload_as_protocol(tmp_path) -> None:
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    bad_path = tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / "bad.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json")

    with pytest.raises(json.JSONDecodeError):
        pool_worker_main(
            tmp_path,
            DEFAULT_SPEC_KEY,
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
        )

    failed_path = tmp_path / "queue" / "failed" / "bad.json"
    payload = json.loads(failed_path.read_text())
    assert payload["failure_kind"] == "protocol"


def test_pool_worker_marks_failed_when_state_failed(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    FlakyPoolTask._create_calls = 0
    task = FlakyPoolTask()

    with pytest.raises(RuntimeError, match="boom"):
        task.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)

    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    payload = {
        "hash": task.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": task.to_dict(),
    }
    todo_path = (
        tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / f"{task.furu_hash}.json"
    )
    todo_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(furu.FuruComputeError, match="already failed"):
        pool_worker_main(
            tmp_path,
            DEFAULT_SPEC_KEY,
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
        )

    failed_path = tmp_path / "queue" / "failed" / f"{task.furu_hash}.json"
    done_path = tmp_path / "queue" / "done" / f"{task.furu_hash}.json"
    assert failed_path.exists()
    assert not done_path.exists()
    payload = json.loads(failed_path.read_text())
    assert payload["failure_kind"] == "compute"


def test_run_slurm_pool_uses_task_executor_spec(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    task = GpuPoolTask(value=1)
    seen_spec_keys: list[str] = []

    def fake_make_executor(
        spec_key: str,
        spec: SlurmSpec,
        *,
        kind: str,
        submitit_root,
        run_id: str | None = None,
    ):
        seen_spec_keys.append(spec_key)
        return InlineExecutor()

    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        fake_make_executor,
    )

    run_slurm_pool(
        [task],
        max_workers_total=1,
        window_size="dfs",
        idle_timeout_sec=0.01,
        poll_interval_sec=0.01,
        submitit_root=None,
        run_root=tmp_path,
    )

    assert task.exists()
    assert seen_spec_keys
    assert seen_spec_keys[0] == GPU_SPEC_KEY


def test_run_slurm_pool_fails_on_failed_queue(tmp_path, monkeypatch) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _ensure_queue_layout(run_dir, specs)
    failed_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
        "error": "boom",
        "failure_kind": "protocol",
        "attempt": 1,
    }
    failed_path = run_dir / "queue" / "failed" / f"{root.furu_hash}.json"
    failed_path.write_text(json.dumps(failed_payload, indent=2))

    def fake_run_dir(_: object) -> Path:
        return run_dir

    monkeypatch.setattr("furu.execution.slurm_pool._run_dir", fake_run_dir)

    with pytest.raises(RuntimeError, match="Protocol failure"):
        run_slurm_pool(
            [root],
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            submitit_root=None,
            run_root=tmp_path,
        )


def test_run_slurm_pool_requeues_stale_running(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _ensure_queue_layout(run_dir, specs)
    running_path = (
        run_dir
        / "queue"
        / "running"
        / DEFAULT_SPEC_KEY
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
        "attempt": 1,
    }
    running_path.write_text(json.dumps(running_payload, indent=2))
    os.utime(running_path, (time.time() - 10_000, time.time() - 10_000))
    hb_path = running_path.with_suffix(".hb")
    hb_path.write_text("alive")
    os.utime(hb_path, (time.time() - 10_000, time.time() - 10_000))

    def fake_run_dir(_: object) -> Path:
        return run_dir

    def fake_build_plan(roots, *, completed_hashes=None):
        return DependencyPlan(
            roots=[root],
            nodes={
                root.furu_hash: PlanNode(
                    obj=root,
                    status="DONE",
                    executor=DEFAULT_SPEC,
                    executor_key=DEFAULT_SPEC_KEY,
                    deps_all=set(),
                    deps_pending=set(),
                    dependents=set(),
                )
            },
        )

    monkeypatch.setattr("furu.execution.slurm_pool._run_dir", fake_run_dir)
    monkeypatch.setattr("furu.execution.slurm_pool.build_plan", fake_build_plan)
    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        lambda *args, **kwargs: NoopExecutor(),
    )

    run_slurm_pool(
        [root],
        max_workers_total=1,
        window_size="dfs",
        idle_timeout_sec=0.01,
        poll_interval_sec=0.01,
        stale_running_sec=1.0,
        submitit_root=None,
        run_root=tmp_path,
    )

    todo_path = run_dir / "queue" / "todo" / DEFAULT_SPEC_KEY / f"{root.furu_hash}.json"
    assert todo_path.exists()
    assert not running_path.exists()


def test_handle_failed_tasks_clears_stale_metadata(tmp_path) -> None:
    root = PoolTask(value=3)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    failed_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
        "error": "boom",
        "failure_kind": "compute",
        "attempt": 1,
        "worker_id": "worker-1",
        "traceback": "Traceback: boom",
        "failed_at": "2026-01-23T00:00:00Z",
    }
    failed_path = tmp_path / "queue" / "failed" / f"{root.furu_hash}.json"
    failed_path.write_text(json.dumps(failed_payload, indent=2))

    entries = _scan_failed_tasks(tmp_path)
    requeued = _handle_failed_tasks(
        tmp_path,
        entries,
        retry_failed=True,
        max_compute_retries=3,
    )

    assert requeued == 1
    assert not failed_path.exists()

    todo_path = (
        tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / f"{root.furu_hash}.json"
    )
    payload = json.loads(todo_path.read_text())
    assert payload["attempt"] == 2
    assert payload["hash"] == root.furu_hash
    assert payload["spec_key"] == DEFAULT_SPEC_KEY
    assert payload["obj"] == root.to_dict()
    assert "error" not in payload
    assert "failure_kind" not in payload
    assert "worker_id" not in payload
    assert "traceback" not in payload
    assert "failed_at" not in payload


def test_handle_failed_tasks_requeues_on_max_retry(tmp_path) -> None:
    root = PoolTask(value=3)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    failed_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
        "error": "boom",
        "failure_kind": "compute",
        "attempt": 3,
        "worker_id": "worker-1",
        "traceback": "Traceback: boom",
        "failed_at": "2026-01-23T00:00:00Z",
    }
    failed_path = tmp_path / "queue" / "failed" / f"{root.furu_hash}.json"
    failed_path.write_text(json.dumps(failed_payload, indent=2))

    entries = _scan_failed_tasks(tmp_path)
    requeued = _handle_failed_tasks(
        tmp_path,
        entries,
        retry_failed=True,
        max_compute_retries=3,
    )

    assert requeued == 1
    todo_path = (
        tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / f"{root.furu_hash}.json"
    )
    payload = json.loads(todo_path.read_text())
    assert payload["attempt"] == 4


def test_handle_failed_tasks_stops_after_retries_exhausted(tmp_path) -> None:
    root = PoolTask(value=3)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    failed_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
        "error": "boom",
        "failure_kind": "compute",
        "attempt": 4,
        "worker_id": "worker-1",
        "traceback": "Traceback: boom",
        "failed_at": "2026-01-23T00:00:00Z",
    }
    failed_path = tmp_path / "queue" / "failed" / f"{root.furu_hash}.json"
    failed_path.write_text(json.dumps(failed_payload, indent=2))

    entries = _scan_failed_tasks(tmp_path)

    with pytest.raises(RuntimeError, match="exhausted retries"):
        _handle_failed_tasks(
            tmp_path,
            entries,
            retry_failed=True,
            max_compute_retries=3,
        )


def test_claim_task_updates_mtime_for_heartbeat_grace(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    todo_path = (
        tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / f"{root.furu_hash}.json"
    )
    todo_path.write_text(
        json.dumps(
            {
                "hash": root.furu_hash,
                "spec_key": DEFAULT_SPEC_KEY,
                "obj": root.to_dict(),
                "attempt": 1,
            },
            indent=2,
        )
    )
    old_time = time.time() - 10_000
    os.utime(todo_path, (old_time, old_time))

    task_path = _claim_task(tmp_path, DEFAULT_SPEC_KEY, "worker-1")

    assert task_path is not None
    assert task_path.exists()
    assert time.time() - task_path.stat().st_mtime < 2.0

    moved = _requeue_stale_running(
        tmp_path,
        stale_sec=60.0,
        heartbeat_grace_sec=1.0,
        max_compute_retries=3,
    )

    assert moved == 0
    assert task_path.exists()


def test_claim_task_ignores_temp_files(tmp_path) -> None:
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    todo_dir = tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY
    # Simulate an in-progress atomic write (or leftover tmp file) which should
    # never be claimed as a task.
    tmp_file = todo_dir / "deadbeef.json.tmp-123"
    tmp_file.write_text("{}")

    task_path = _claim_task(tmp_path, DEFAULT_SPEC_KEY, "worker-1")

    assert task_path is None
    assert tmp_file.exists()


def test_requeue_stale_running_respects_heartbeat(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / DEFAULT_SPEC_KEY
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
    }
    running_path.write_text(json.dumps(running_payload, indent=2))
    os.utime(running_path, (time.time() - 10_000, time.time() - 10_000))
    hb_path = running_path.with_suffix(".hb")
    hb_path.write_text("alive")
    os.utime(hb_path, None)

    moved = _requeue_stale_running(
        tmp_path,
        stale_sec=0.01,
        heartbeat_grace_sec=1.0,
        max_compute_retries=3,
    )

    assert moved == 0
    assert running_path.exists()
    assert not (
        tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / running_path.name
    ).exists()


def test_requeue_stale_running_invalid_claimed_at_does_not_crash(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / DEFAULT_SPEC_KEY
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
        "attempt": 1,
        "claimed_at": "not-a-timestamp",
    }
    running_path.write_text(json.dumps(running_payload, indent=2))
    old_time = time.time() - 10_000
    os.utime(running_path, (old_time, old_time))

    moved = _requeue_stale_running(
        tmp_path,
        stale_sec=60.0,
        heartbeat_grace_sec=0.01,
        max_compute_retries=3,
    )

    assert moved == 1
    todo_path = (
        tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / f"{root.furu_hash}.json"
    )
    assert todo_path.exists()
    assert not running_path.exists()


def test_requeue_stale_running_missing_heartbeat_grace(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / DEFAULT_SPEC_KEY
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
    }
    running_path.write_text(json.dumps(running_payload, indent=2))
    os.utime(running_path, None)

    moved = _requeue_stale_running(
        tmp_path,
        stale_sec=0.01,
        heartbeat_grace_sec=60.0,
        max_compute_retries=3,
    )

    assert moved == 0
    assert running_path.exists()
    assert not (tmp_path / "queue" / "failed" / running_path.name).exists()


def test_requeue_stale_running_missing_heartbeat_requeues_once(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / DEFAULT_SPEC_KEY
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
    }
    running_path.write_text(json.dumps(running_payload, indent=2))
    stale_time = time.time() - 120
    os.utime(running_path, (stale_time, stale_time))

    moved = _requeue_stale_running(
        tmp_path,
        stale_sec=0.01,
        heartbeat_grace_sec=1.0,
        max_compute_retries=3,
    )

    todo_path = tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / running_path.name
    payload = json.loads(todo_path.read_text())

    assert moved == 1
    assert not running_path.exists()
    assert payload["missing_heartbeat_requeues"] == 1
    assert not (tmp_path / "queue" / "failed" / running_path.name).exists()


def test_requeue_stale_running_missing_heartbeat_exhausts(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / DEFAULT_SPEC_KEY
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
        "missing_heartbeat_requeues": 1,
    }
    running_path.write_text(json.dumps(running_payload, indent=2))
    stale_time = time.time() - 120
    os.utime(running_path, (stale_time, stale_time))

    moved = _requeue_stale_running(
        tmp_path,
        stale_sec=0.01,
        heartbeat_grace_sec=1.0,
        max_compute_retries=3,
    )

    failed_path = tmp_path / "queue" / "failed" / running_path.name
    payload = json.loads(failed_path.read_text())

    assert moved == 0
    assert not running_path.exists()
    assert payload["failure_kind"] == "protocol"
    assert payload["missing_heartbeat_requeues"] == 1


def test_requeue_stale_running_bounds_attempts(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / DEFAULT_SPEC_KEY
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": DEFAULT_SPEC_KEY,
        "obj": root.to_dict(),
        "attempt": 1,
    }
    running_path.write_text(json.dumps(running_payload, indent=2))
    hb_path = running_path.with_suffix(".hb")
    hb_path.write_text("alive")
    stale_time = time.time() - 120
    os.utime(hb_path, (stale_time, stale_time))

    moved = _requeue_stale_running(
        tmp_path,
        stale_sec=0.01,
        heartbeat_grace_sec=1.0,
        max_compute_retries=1,
    )

    todo_path = tmp_path / "queue" / "todo" / DEFAULT_SPEC_KEY / running_path.name
    payload = json.loads(todo_path.read_text())

    assert moved == 1
    assert payload["attempt"] == 2
    assert payload["stale_heartbeat_requeues"] == 1

    todo_path.replace(running_path)
    hb_path.write_text("alive")
    os.utime(hb_path, (stale_time, stale_time))

    with pytest.raises(RuntimeError, match="Stale heartbeat exhausted retries"):
        _requeue_stale_running(
            tmp_path,
            stale_sec=0.01,
            heartbeat_grace_sec=1.0,
            max_compute_retries=1,
        )

    failed_path = tmp_path / "queue" / "failed" / running_path.name
    payload = json.loads(failed_path.read_text())
    assert payload["failure_kind"] == "protocol"


def test_mark_done_handles_missing_task_path(tmp_path) -> None:
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)
    missing_path = (
        tmp_path / "queue" / "running" / DEFAULT_SPEC_KEY / "worker-1" / "missing.json"
    )
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    hb_path = missing_path.with_suffix(".hb")
    hb_path.write_text("alive")

    _mark_done(tmp_path, missing_path)

    assert not hb_path.exists()
    assert not (tmp_path / "queue" / "done" / missing_path.name).exists()


def test_mark_failed_handles_invalid_payload(tmp_path) -> None:
    specs = {DEFAULT_SPEC_KEY: DEFAULT_SPEC}
    _ensure_queue_layout(tmp_path, specs)
    task_path = (
        tmp_path / "queue" / "running" / DEFAULT_SPEC_KEY / "worker-1" / "bad.json"
    )
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text("{not-json")
    hb_path = task_path.with_suffix(".hb")
    hb_path.write_text("alive")

    _mark_failed(tmp_path, task_path, "boom", failure_kind="protocol")

    failed_path = tmp_path / "queue" / "failed" / task_path.name
    payload = json.loads(failed_path.read_text())
    assert payload["error"] == "boom"
    assert payload["hash"] == "bad"
    assert payload["failure_kind"] == "protocol"
    assert payload["attempt"] == 1
    assert not hb_path.exists()


def test_run_slurm_pool_detects_no_progress(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    root = PoolTask(value=1)
    blocked = PoolTask(value=2)
    plan = DependencyPlan(
        roots=[root],
        nodes={
            root.furu_hash: PlanNode(
                obj=root,
                status="TODO",
                executor=DEFAULT_SPEC,
                executor_key=DEFAULT_SPEC_KEY,
                deps_all={blocked.furu_hash},
                deps_pending={blocked.furu_hash},
                dependents=set(),
            ),
            blocked.furu_hash: PlanNode(
                obj=blocked,
                status="TODO",
                executor=DEFAULT_SPEC,
                executor_key=DEFAULT_SPEC_KEY,
                deps_all={root.furu_hash},
                deps_pending={root.furu_hash},
                dependents=set(),
            ),
        },
    )

    def fake_build_plan(roots, *, completed_hashes=None):
        return plan

    monkeypatch.setattr("furu.execution.slurm_pool.build_plan", fake_build_plan)

    with pytest.raises(RuntimeError, match="no progress"):
        run_slurm_pool(
            [root],
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            submitit_root=None,
            run_root=tmp_path,
        )


def test_run_slurm_pool_stale_in_progress_raises(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    root = PoolTask(value=1)
    directory = root._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)
    furu.StateManager.start_attempt_running(
        directory,
        backend="submitit",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    lock_path = furu.StateManager.get_lock_path(
        directory, furu.StateManager.COMPUTE_LOCK
    )
    lock_path.write_text("lock")
    stale_time = time.time() - 5.0
    os.utime(lock_path, (stale_time, stale_time))

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)
    monkeypatch.setattr(furu.FURU_CONFIG, "stale_timeout", 0.01)

    with pytest.raises(RuntimeError, match="Stale IN_PROGRESS dependencies detected"):
        run_slurm_pool(
            [root],
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            submitit_root=None,
            run_root=tmp_path,
        )


def test_run_slurm_pool_throttles_plan_refresh(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    root = PoolTask(value=1)
    build_calls = 0

    plan = DependencyPlan(
        roots=[root],
        nodes={
            root.furu_hash: PlanNode(
                obj=root,
                status="IN_PROGRESS",
                executor=DEFAULT_SPEC,
                executor_key=DEFAULT_SPEC_KEY,
                deps_all=set(),
                deps_pending=set(),
                dependents=set(),
            )
        },
    )

    def counted_build_plan(roots, *, completed_hashes=None):
        nonlocal build_calls
        build_calls += 1
        return plan

    sleep_calls = 0

    def stop_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise RuntimeError("stop loop")

    monkeypatch.setattr("furu.execution.slurm_pool.build_plan", counted_build_plan)
    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        lambda *args, **kwargs: NoopExecutor(),
    )
    monkeypatch.setattr(
        "furu.execution.slurm_pool.reconcile_or_timeout_in_progress",
        lambda plan, *, stale_timeout_sec: False,
    )
    monkeypatch.setattr("furu.execution.slurm_pool.time.sleep", stop_sleep)

    with pytest.raises(RuntimeError, match="stop loop"):
        run_slurm_pool(
            [root],
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            plan_refresh_interval_sec=60.0,
            submitit_root=None,
            run_root=tmp_path,
        )

    assert build_calls <= 2


def test_run_slurm_pool_throttles_worker_health_checks(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    root = PoolTask(value=1)
    job = CountingJob()
    executor = CountingExecutor(job)

    sleep_calls = 0

    def stop_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise RuntimeError("stop loop")

    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        lambda *args, **kwargs: executor,
    )
    monkeypatch.setattr("furu.execution.slurm_pool.time.sleep", stop_sleep)

    with pytest.raises(RuntimeError, match="stop loop"):
        run_slurm_pool(
            [root],
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            worker_health_check_interval_sec=60.0,
            submitit_root=None,
            run_root=tmp_path,
        )

    assert executor.submitted == 1
    assert job.done_calls <= 1


def test_run_slurm_pool_throttles_queue_scans(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    root = PoolTask(value=1)
    plan = DependencyPlan(
        roots=[root],
        nodes={
            root.furu_hash: PlanNode(
                obj=root,
                status="IN_PROGRESS",
                executor=DEFAULT_SPEC,
                executor_key=DEFAULT_SPEC_KEY,
                deps_all=set(),
                deps_pending=set(),
                dependents=set(),
            )
        },
    )

    sleep_calls = 0
    done_scan_calls = 0
    failed_scan_calls = 0
    stale_scan_calls = 0

    def stop_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise RuntimeError("stop loop")

    def counted_done_hashes(_run_dir: Path) -> set[str]:
        nonlocal done_scan_calls
        done_scan_calls += 1
        return set()

    def counted_failed_scan(_run_dir: Path):
        nonlocal failed_scan_calls
        failed_scan_calls += 1
        return []

    def counted_stale_scan(
        _run_dir: Path,
        *,
        stale_sec: float,
        heartbeat_grace_sec: float,
        max_compute_retries: int,
    ) -> int:
        nonlocal stale_scan_calls
        stale_scan_calls += 1
        return 0

    monkeypatch.setattr("furu.execution.slurm_pool.build_plan", lambda *a, **k: plan)
    monkeypatch.setattr("furu.execution.slurm_pool._done_hashes", counted_done_hashes)
    monkeypatch.setattr(
        "furu.execution.slurm_pool._scan_failed_tasks",
        counted_failed_scan,
    )
    monkeypatch.setattr(
        "furu.execution.slurm_pool._requeue_stale_running",
        counted_stale_scan,
    )
    monkeypatch.setattr(
        "furu.execution.slurm_pool.make_executor_for_spec",
        lambda *args, **kwargs: NoopExecutor(),
    )
    monkeypatch.setattr(
        "furu.execution.slurm_pool.reconcile_or_timeout_in_progress",
        lambda plan, *, stale_timeout_sec: False,
    )
    monkeypatch.setattr("furu.execution.slurm_pool.time.sleep", stop_sleep)

    with pytest.raises(RuntimeError, match="stop loop"):
        run_slurm_pool(
            [root],
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            done_scan_interval_sec=60.0,
            failed_scan_interval_sec=60.0,
            stale_scan_interval_sec=60.0,
            submitit_root=None,
            run_root=tmp_path,
        )

    assert done_scan_calls <= 2
    assert failed_scan_calls <= 1
    assert stale_scan_calls <= 1
