import json
import os
import time
from typing import ClassVar

import pytest

import furu
from furu import furu_dep
from furu.execution.local import run_local
from furu.execution.plan import DependencyPlan, PlanNode
from furu.execution.slurm_spec import SlurmSpec, slurm_spec_key


DEFAULT_EXECUTOR = SlurmSpec()
DEFAULT_EXECUTOR_KEY = slurm_spec_key(DEFAULT_EXECUTOR)


class LeafTask(furu.Furu[int]):
    value: int = furu.chz.field()
    _create_calls: int = 0
    _load_calls: int = 0

    def _create(self) -> int:
        object.__setattr__(self, "_create_calls", self._create_calls + 1)
        (self.furu_dir / "value.json").write_text(json.dumps(self.value))
        return self.value

    def _load(self) -> int:
        object.__setattr__(self, "_load_calls", self._load_calls + 1)
        return json.loads((self.furu_dir / "value.json").read_text())


class SumTask(furu.Furu[int]):
    deps: list[furu.Furu[int]] = furu.chz.field(default_factory=list)

    @furu_dep
    def task_dependencies(self) -> furu.DependencySpec:
        return self.deps

    def _create(self) -> int:
        total = sum(dep.get() for dep in self.deps)
        (self.furu_dir / "sum.json").write_text(json.dumps(total))
        return total

    def _load(self) -> int:
        return json.loads((self.furu_dir / "sum.json").read_text())


class FailingTask(furu.Furu[int]):
    def _create(self) -> int:
        raise ValueError("boom")

    def _load(self) -> int:
        raise AssertionError("should not load")


class FlakyExecutorTask(furu.Furu[int]):
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


class FlakyNTimesTask(furu.Furu[int]):
    fail_times: ClassVar[int] = 0
    _create_calls: ClassVar[int] = 0

    def _create(self) -> int:
        type(self)._create_calls += 1
        if type(self)._create_calls <= type(self).fail_times:
            raise RuntimeError("boom")
        value = 7
        (self.furu_dir / "value.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


def test_run_local_executes_dependencies(furu_tmp_root) -> None:
    leaf = LeafTask(value=2)
    root = SumTask(deps=[leaf])

    run_local([root], max_workers=2, window_size="bfs", poll_interval_sec=0.01)

    assert leaf._create_calls == 1
    assert leaf._load_calls == 1
    assert root.get() == 2


def test_run_local_fails_fast_on_error(furu_tmp_root, monkeypatch) -> None:
    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)
    with pytest.raises(furu.FuruComputeError):
        run_local(
            [FailingTask()], max_workers=1, window_size="dfs", poll_interval_sec=0.01
        )


def test_run_local_always_rerun_recomputes(furu_tmp_root, monkeypatch) -> None:
    task = LeafTask(value=2)
    task.get()

    qualname = f"{LeafTask.__module__}.{LeafTask.__qualname__}"
    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun_all", False)
    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun", {qualname})

    run_local([task], max_workers=1, window_size="dfs", poll_interval_sec=0.01)

    assert task._create_calls == 2


def test_run_local_retries_failed_when_enabled(furu_tmp_root, monkeypatch) -> None:
    FlakyExecutorTask._create_calls = 0
    task = FlakyExecutorTask()

    with pytest.raises(RuntimeError, match="boom"):
        task.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    run_local([task], max_workers=1, window_size="dfs", poll_interval_sec=0.01)

    assert task.get() == 7
    assert FlakyExecutorTask._create_calls == 2


def test_run_local_retries_compute_failures(furu_tmp_root, monkeypatch) -> None:
    FlakyExecutorTask._create_calls = 0
    task = FlakyExecutorTask()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    monkeypatch.setattr(furu.FURU_CONFIG, "max_compute_retries", 2)

    run_local([task], max_workers=1, window_size="dfs", poll_interval_sec=0.01)

    assert task.get() == 7
    assert FlakyExecutorTask._create_calls == 2


def test_run_local_allows_max_compute_retries(furu_tmp_root, monkeypatch) -> None:
    FlakyNTimesTask.fail_times = 3
    FlakyNTimesTask._create_calls = 0
    task = FlakyNTimesTask()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    monkeypatch.setattr(furu.FURU_CONFIG, "max_compute_retries", 3)

    run_local([task], max_workers=1, window_size="dfs", poll_interval_sec=0.01)

    assert task.get() == 7
    assert FlakyNTimesTask._create_calls == 4


def test_run_local_respects_zero_compute_retries(furu_tmp_root, monkeypatch) -> None:
    FlakyNTimesTask.fail_times = 1
    FlakyNTimesTask._create_calls = 0
    task = FlakyNTimesTask()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    monkeypatch.setattr(furu.FURU_CONFIG, "max_compute_retries", 0)

    with pytest.raises(furu.FuruComputeError):
        run_local([task], max_workers=1, window_size="dfs", poll_interval_sec=0.01)


def test_run_local_fails_fast_on_failed_state_when_retry_disabled(
    furu_tmp_root, monkeypatch
) -> None:
    task = FailingTask()

    with pytest.raises(ValueError, match="boom"):
        task.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)
    with pytest.raises(RuntimeError, match="failed dependencies"):
        run_local([task], max_workers=1, window_size="dfs", poll_interval_sec=0.01)


def test_run_local_detects_no_progress(furu_tmp_root, monkeypatch) -> None:
    root = LeafTask(value=1)
    blocked = LeafTask(value=2)
    plan = DependencyPlan(
        roots=[root],
        nodes={
            root.furu_hash: PlanNode(
                obj=root,
                status="TODO",
                executor=DEFAULT_EXECUTOR,
                executor_key=DEFAULT_EXECUTOR_KEY,
                deps_all={blocked.furu_hash},
                deps_pending={blocked.furu_hash},
                dependents=set(),
            ),
            blocked.furu_hash: PlanNode(
                obj=blocked,
                status="TODO",
                executor=DEFAULT_EXECUTOR,
                executor_key=DEFAULT_EXECUTOR_KEY,
                deps_all={root.furu_hash},
                deps_pending={root.furu_hash},
                dependents=set(),
            ),
        },
    )

    def fake_build_plan(roots, *, completed_hashes=None):
        return plan

    monkeypatch.setattr("furu.execution.local.build_plan", fake_build_plan)

    with pytest.raises(RuntimeError, match="no progress"):
        run_local([root], max_workers=1, window_size="dfs", poll_interval_sec=0.01)


def test_run_local_reconciles_stale_in_progress(furu_tmp_root, monkeypatch) -> None:
    task = LeafTask(value=3)
    directory = task._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)
    furu.StateManager.start_attempt_running(
        directory,
        backend="local",
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

    monkeypatch.setattr(furu.FURU_CONFIG, "stale_timeout", 0.01)

    run_local([task], max_workers=1, window_size="dfs", poll_interval_sec=0.01)

    assert task._create_calls == 1
