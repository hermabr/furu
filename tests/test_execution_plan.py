import json
import os
import time

import furu
from furu.execution.plan import (
    build_plan,
    ready_todo,
    reconcile_in_progress,
    topo_order_todo,
)


class Task(furu.Furu[int]):
    name: str = furu.chz.field()
    deps: list[furu.Furu[int]] = furu.chz.field(default_factory=list)

    def _dependencies(self) -> furu.DependencySpec:
        return self.deps

    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(self.name))
        return 1

    def _load(self) -> int:
        json.loads((self.furu_dir / "value.json").read_text())
        return 1


class InvalidValidateTask(Task):
    def _validate(self) -> bool:
        raise furu.FuruValidationError("invalid")


class ExplodingValidateTask(Task):
    def _validate(self) -> bool:
        raise RuntimeError("validate error")


def test_build_plan_classifies_statuses(furu_tmp_root) -> None:
    done = Task(name="done")
    done.get()

    in_progress = Task(name="in-progress")
    directory = in_progress._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)
    furu.StateManager.start_attempt_queued(
        directory,
        backend="local",
        lease_duration_sec=0.05,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    root = Task(name="root", deps=[done, in_progress])
    plan = build_plan([root])

    assert plan.nodes[root.furu_hash].status == "TODO"
    assert plan.nodes[done.furu_hash].status == "DONE"
    assert plan.nodes[in_progress.furu_hash].status == "IN_PROGRESS"
    assert plan.nodes[root.furu_hash].deps_pending == {in_progress.furu_hash}
    assert ready_todo(plan) == []


def test_build_plan_prunes_completed_subgraphs(furu_tmp_root) -> None:
    leaf = Task(name="leaf")
    mid = Task(name="mid", deps=[leaf])
    mid.get()

    root = Task(name="root", deps=[mid])
    plan = build_plan([root])

    assert root.furu_hash in plan.nodes
    assert mid.furu_hash in plan.nodes
    assert leaf.furu_hash not in plan.nodes
    assert ready_todo(plan) == [root.furu_hash]


def test_build_plan_treats_validation_error_as_missing(furu_tmp_root, caplog) -> None:
    task = InvalidValidateTask(name="bad")
    directory = task._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)
    attempt_id = furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=0.05,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )
    furu.StateManager.finish_attempt_success(directory, attempt_id=attempt_id)
    furu.StateManager.write_success_marker(directory, attempt_id=attempt_id)

    caplog.set_level("WARNING")
    plan = build_plan([task])

    assert plan.nodes[task.furu_hash].status == "TODO"
    assert any("validate invalid" in record.getMessage() for record in caplog.records)


def test_build_plan_logs_unexpected_validate_error(furu_tmp_root, caplog) -> None:
    task = ExplodingValidateTask(name="boom")
    directory = task._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)
    attempt_id = furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=0.05,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )
    furu.StateManager.finish_attempt_success(directory, attempt_id=attempt_id)
    furu.StateManager.write_success_marker(directory, attempt_id=attempt_id)

    caplog.set_level("ERROR")
    plan = build_plan([task])

    assert plan.nodes[task.furu_hash].status == "TODO"
    assert any("validate crashed" in record.getMessage() for record in caplog.records)


def test_topo_order_todo_returns_dependency_order(furu_tmp_root) -> None:
    leaf = Task(name="leaf")
    mid = Task(name="mid", deps=[leaf])
    root = Task(name="root", deps=[mid])

    plan = build_plan([root])

    assert topo_order_todo(plan) == [leaf.furu_hash, mid.furu_hash, root.furu_hash]


def test_build_plan_marks_failed_attempts(furu_tmp_root, monkeypatch) -> None:
    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)
    failed = Task(name="failed")
    directory = failed._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)
    attempt_id = furu.StateManager.start_attempt_running(
        directory,
        backend="local",
        lease_duration_sec=0.05,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )
    furu.StateManager.finish_attempt_failed(
        directory,
        attempt_id=attempt_id,
        error={"type": "ValueError", "message": "boom"},
    )

    plan = build_plan([failed])

    assert plan.nodes[failed.furu_hash].status == "FAILED"


def test_build_plan_does_not_log_exists(furu_tmp_root, caplog) -> None:
    root = Task(name="root")

    caplog.set_level("INFO")
    build_plan([root])

    assert all("exists" not in record.getMessage() for record in caplog.records)


def test_reconcile_in_progress_skips_fresh_attempts(furu_tmp_root) -> None:
    task = Task(name="in-progress")
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

    plan = build_plan([task])

    assert reconcile_in_progress(plan, stale_timeout_sec=9999.0) is False
    state = furu.StateManager.read_state(directory)
    assert state.attempt is not None
    assert state.attempt.status in {"queued", "running"}


def test_reconcile_in_progress_missing_timestamps_not_immediately_stale(
    furu_tmp_root, monkeypatch
) -> None:
    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    task = Task(name="missing-timestamps")
    directory = task._base_furu_dir()
    furu.StateManager.ensure_internal_dir(directory)
    furu.StateManager.start_attempt_queued(
        directory,
        backend="local",
        lease_duration_sec=60.0,
        owner={"pid": 99999, "host": "other-host", "user": "x"},
        scheduler={},
    )

    def mutate(state) -> None:
        attempt = state.attempt
        assert attempt is not None
        attempt.started_at = ""

    furu.StateManager.update_state(directory, mutate)
    plan = build_plan([task])

    assert reconcile_in_progress(plan, stale_timeout_sec=5.0) is False
    state = furu.StateManager.read_state(directory)
    assert state.attempt is not None
    assert state.attempt.status == "queued"


def test_reconcile_in_progress_stale_heartbeat_marks_crashed(
    furu_tmp_root, monkeypatch
) -> None:
    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    task = Task(name="stale-attempt")
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
    stale_time = time.time() - 120.0
    os.utime(lock_path, (stale_time, stale_time))
    plan = build_plan([task])

    assert reconcile_in_progress(plan, stale_timeout_sec=1.0) is False
    state = furu.StateManager.read_state(directory)
    assert state.attempt is not None
    assert state.attempt.status == "crashed"
    assert getattr(state.attempt, "reason", None) == "lease_expired"
