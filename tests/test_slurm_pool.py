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
    _handle_failed_tasks,
    _mark_done,
    _mark_failed,
    _requeue_stale_running,
    _scan_failed_tasks,
    _ensure_queue_layout,
    pool_worker_main,
    run_slurm_pool,
)
from furu.execution.slurm_spec import SlurmSpec


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
    def _executor_spec_key(self) -> str:
        return "gpu"


def test_run_slurm_pool_executes_tasks(furu_tmp_root, tmp_path, monkeypatch) -> None:
    root = PoolTask(value=3)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

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
        specs=specs,
        max_workers_total=1,
        window_size="dfs",
        idle_timeout_sec=0.01,
        poll_interval_sec=0.01,
        submitit_root=None,
        run_root=tmp_path,
    )

    assert root.exists()
    assert (run.run_dir / "queue" / "done" / f"{root.furu_hash}.json").exists()


def test_run_slurm_pool_retries_failed_when_enabled(
    furu_tmp_root, tmp_path, monkeypatch
) -> None:
    FlakyPoolTask._create_calls = 0
    root = FlakyPoolTask()
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

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
        specs=specs,
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
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

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
            specs=specs,
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            submitit_root=None,
            run_root=tmp_path,
        )


def test_pool_worker_detects_spec_mismatch(tmp_path) -> None:
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    task = GpuPoolTask(value=1)
    payload = {
        "hash": task.furu_hash,
        "spec_key": "default",
        "obj": task.to_dict(),
    }
    todo_path = tmp_path / "queue" / "todo" / "default" / f"{task.furu_hash}.json"
    todo_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(RuntimeError):
        pool_worker_main(
            tmp_path,
            "default",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
        )

    failed_path = tmp_path / "queue" / "failed" / f"{task.furu_hash}.json"
    assert failed_path.exists()
    payload = json.loads(failed_path.read_text())
    assert payload["failure_kind"] == "protocol"


def test_pool_worker_marks_invalid_json_payload_as_protocol(tmp_path) -> None:
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    bad_path = tmp_path / "queue" / "todo" / "default" / "bad.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json")

    with pytest.raises(json.JSONDecodeError):
        pool_worker_main(
            tmp_path,
            "default",
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

    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    payload = {
        "hash": task.furu_hash,
        "spec_key": "default",
        "obj": task.to_dict(),
    }
    todo_path = tmp_path / "queue" / "todo" / "default" / f"{task.furu_hash}.json"
    todo_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(furu.FuruComputeError, match="already failed"):
        pool_worker_main(
            tmp_path,
            "default",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
        )

    failed_path = tmp_path / "queue" / "failed" / f"{task.furu_hash}.json"
    done_path = tmp_path / "queue" / "done" / f"{task.furu_hash}.json"
    assert failed_path.exists()
    assert not done_path.exists()
    payload = json.loads(failed_path.read_text())
    assert payload["failure_kind"] == "compute"


def test_run_slurm_pool_missing_spec_key_raises(furu_tmp_root, tmp_path) -> None:
    task = GpuPoolTask(value=1)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

    with pytest.raises(KeyError, match="gpu"):
        run_slurm_pool(
            [task],
            specs=specs,
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            submitit_root=None,
            run_root=tmp_path,
        )


def test_run_slurm_pool_fails_on_failed_queue(tmp_path, monkeypatch) -> None:
    root = PoolTask(value=1)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _ensure_queue_layout(run_dir, specs)
    failed_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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
            specs=specs,
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
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _ensure_queue_layout(run_dir, specs)
    running_path = (
        run_dir
        / "queue"
        / "running"
        / "default"
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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
                    spec_key="default",
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
        specs=specs,
        max_workers_total=1,
        window_size="dfs",
        idle_timeout_sec=0.01,
        poll_interval_sec=0.01,
        stale_running_sec=1.0,
        submitit_root=None,
        run_root=tmp_path,
    )

    todo_path = run_dir / "queue" / "todo" / "default" / f"{root.furu_hash}.json"
    assert todo_path.exists()
    assert not running_path.exists()


def test_handle_failed_tasks_clears_stale_metadata(tmp_path) -> None:
    root = PoolTask(value=3)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    failed_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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

    todo_path = tmp_path / "queue" / "todo" / "default" / f"{root.furu_hash}.json"
    payload = json.loads(todo_path.read_text())
    assert payload["attempt"] == 2
    assert payload["hash"] == root.furu_hash
    assert payload["spec_key"] == "default"
    assert payload["obj"] == root.to_dict()
    assert "error" not in payload
    assert "failure_kind" not in payload
    assert "worker_id" not in payload
    assert "traceback" not in payload
    assert "failed_at" not in payload


def test_handle_failed_tasks_requeues_on_max_retry(tmp_path) -> None:
    root = PoolTask(value=3)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    failed_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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
    todo_path = tmp_path / "queue" / "todo" / "default" / f"{root.furu_hash}.json"
    payload = json.loads(todo_path.read_text())
    assert payload["attempt"] == 4


def test_handle_failed_tasks_stops_after_retries_exhausted(tmp_path) -> None:
    root = PoolTask(value=3)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    failed_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    todo_path = tmp_path / "queue" / "todo" / "default" / f"{root.furu_hash}.json"
    todo_path.write_text(
        json.dumps(
            {
                "hash": root.furu_hash,
                "spec_key": "default",
                "obj": root.to_dict(),
                "attempt": 1,
            },
            indent=2,
        )
    )
    old_time = time.time() - 10_000
    os.utime(todo_path, (old_time, old_time))

    task_path = _claim_task(tmp_path, "default", "worker-1")

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
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    todo_dir = tmp_path / "queue" / "todo" / "default"
    # Simulate an in-progress atomic write (or leftover tmp file) which should
    # never be claimed as a task.
    tmp_file = todo_dir / "deadbeef.json.tmp-123"
    tmp_file.write_text("{}")

    task_path = _claim_task(tmp_path, "default", "worker-1")

    assert task_path is None
    assert tmp_file.exists()


def test_requeue_stale_running_respects_heartbeat(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / "default"
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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
    assert not (tmp_path / "queue" / "todo" / "default" / running_path.name).exists()


def test_requeue_stale_running_invalid_claimed_at_does_not_crash(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / "default"
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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
    todo_path = tmp_path / "queue" / "todo" / "default" / f"{root.furu_hash}.json"
    assert todo_path.exists()
    assert not running_path.exists()


def test_requeue_stale_running_missing_heartbeat_grace(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / "default"
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / "default"
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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

    todo_path = tmp_path / "queue" / "todo" / "default" / running_path.name
    payload = json.loads(todo_path.read_text())

    assert moved == 1
    assert not running_path.exists()
    assert payload["missing_heartbeat_requeues"] == 1
    assert not (tmp_path / "queue" / "failed" / running_path.name).exists()


def test_requeue_stale_running_missing_heartbeat_exhausts(tmp_path) -> None:
    root = PoolTask(value=1)
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / "default"
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)

    running_path = (
        tmp_path
        / "queue"
        / "running"
        / "default"
        / "worker-1"
        / f"{root.furu_hash}.json"
    )
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_payload = {
        "hash": root.furu_hash,
        "spec_key": "default",
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

    todo_path = tmp_path / "queue" / "todo" / "default" / running_path.name
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
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)
    missing_path = (
        tmp_path / "queue" / "running" / "default" / "worker-1" / "missing.json"
    )
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    hb_path = missing_path.with_suffix(".hb")
    hb_path.write_text("alive")

    _mark_done(tmp_path, missing_path)

    assert not hb_path.exists()
    assert not (tmp_path / "queue" / "done" / missing_path.name).exists()


def test_mark_failed_handles_invalid_payload(tmp_path) -> None:
    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}
    _ensure_queue_layout(tmp_path, specs)
    task_path = tmp_path / "queue" / "running" / "default" / "worker-1" / "bad.json"
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
                spec_key="default",
                deps_all={blocked.furu_hash},
                deps_pending={blocked.furu_hash},
                dependents=set(),
            ),
            blocked.furu_hash: PlanNode(
                obj=blocked,
                status="TODO",
                spec_key="default",
                deps_all={root.furu_hash},
                deps_pending={root.furu_hash},
                dependents=set(),
            ),
        },
    )

    def fake_build_plan(roots, *, completed_hashes=None):
        return plan

    monkeypatch.setattr("furu.execution.slurm_pool.build_plan", fake_build_plan)

    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

    with pytest.raises(RuntimeError, match="no progress"):
        run_slurm_pool(
            [root],
            specs=specs,
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

    specs = {"default": SlurmSpec(partition="cpu", cpus=2, mem_gb=4, time_min=10)}

    with pytest.raises(RuntimeError, match="Stale IN_PROGRESS dependencies detected"):
        run_slurm_pool(
            [root],
            specs=specs,
            max_workers_total=1,
            window_size="dfs",
            idle_timeout_sec=0.01,
            poll_interval_sec=0.01,
            submitit_root=None,
            run_root=tmp_path,
        )
