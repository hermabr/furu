from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, TypedDict, cast

from ..adapters import SubmititAdapter
from ..adapters.submitit import SubmititJob
from ..config import FURU_CONFIG
from ..core import Furu
from ..errors import FuruComputeError, FuruMissingArtifact, FuruSpecMismatch
from ..runtime.logging import get_logger
from ..serialization.serializer import JsonValue
from ..storage.state import _FuruState, _StateResultFailed, _StateResultSuccess
from .paths import submitit_root_dir
from .plan import DependencyPlan, build_plan, ready_todo
from .plan_utils import reconcile_or_timeout_in_progress
from .slurm_spec import (
    SlurmSpec,
    SlurmSpecExtraValue,
    resolve_executor_specs,
    slurm_spec_key,
)
from .submitit_factory import make_executor_for_spec


FailureKind = Literal["compute", "protocol"]
PoolFailurePhase = Literal["payload", "worker"]
WorkerLimitKey = SlurmSpec | Literal["total"]
WorkerLimitConfig = int | Mapping[WorkerLimitKey, int]
MISSING_HEARTBEAT_REQUEUE_LIMIT = 1


class _TaskPayload(TypedDict, total=False):
    hash: str
    spec_key: str
    executor_keys: list[str]
    obj: JsonValue
    error: str
    traceback: str
    attempt: int
    failure_kind: FailureKind
    failed_at: str
    claimed_at: str
    worker_id: str
    missing_heartbeat_requeues: int
    stale_heartbeat_requeues: int


@dataclass
class SlurmPoolRun:
    run_dir: Path
    submitit_root: Path
    plan: DependencyPlan


@dataclass(frozen=True)
class _WorkerLimits:
    total_limit: int
    per_spec_limits: Mapping[str, int] | None


def classify_pool_exception(
    exc: Exception,
    *,
    phase: PoolFailurePhase,
    state: _FuruState | None = None,
) -> FailureKind:
    if phase == "payload":
        return "protocol"
    if isinstance(exc, (FuruMissingArtifact, FuruSpecMismatch)):
        return "protocol"
    if isinstance(exc, FuruComputeError):
        return "compute"
    if state is not None and isinstance(state.result, _StateResultFailed):
        return "compute"
    return "protocol"


def _normalize_window_size(window_size: str | int, root_count: int) -> int:
    if root_count == 0:
        return 0
    if isinstance(window_size, str):
        match window_size:
            case "dfs":
                return 1
            case "bfs":
                return root_count
            case _:
                raise ValueError(
                    "window_size must be 'dfs', 'bfs', or a positive integer"
                )
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("window_size must be 'dfs', 'bfs', or a positive integer")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    return min(window_size, root_count)


def _normalize_worker_limits(max_workers_total: WorkerLimitConfig) -> _WorkerLimits:
    if isinstance(max_workers_total, bool):
        raise TypeError(
            'max_workers_total must be an int or a mapping of SlurmSpec|"total" to int'
        )
    if isinstance(max_workers_total, int):
        if max_workers_total < 1:
            raise ValueError("max_workers_total must be >= 1")
        return _WorkerLimits(total_limit=max_workers_total, per_spec_limits=None)
    if not isinstance(max_workers_total, Mapping):
        raise TypeError(
            'max_workers_total must be an int or a mapping of SlurmSpec|"total" to int'
        )

    per_spec_limits: dict[str, int] = {}
    total_limit: int | None = None
    for raw_key, raw_limit in max_workers_total.items():
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            raise TypeError("max_workers_total mapping values must be integers >= 1")
        if raw_limit < 1:
            raise ValueError("max_workers_total mapping values must be >= 1")

        if raw_key == "total":
            total_limit = raw_limit
            continue
        if not isinstance(raw_key, SlurmSpec):
            raise TypeError(
                "max_workers_total mapping keys must be SlurmSpec or 'total'"
            )
        per_spec_limits[slurm_spec_key(raw_key)] = raw_limit

    if not per_spec_limits:
        raise ValueError(
            "max_workers_total mapping must include at least one SlurmSpec key"
        )

    resolved_total_limit = total_limit
    if resolved_total_limit is None:
        resolved_total_limit = sum(per_spec_limits.values())

    return _WorkerLimits(
        total_limit=resolved_total_limit,
        per_spec_limits=per_spec_limits,
    )


def _validate_worker_limits_for_plan(
    plan: DependencyPlan,
    *,
    per_spec_limits: Mapping[str, int] | None,
) -> None:
    if per_spec_limits is None:
        return

    missing_keys: set[str] = set()
    for node in plan.nodes.values():
        if node.status == "DONE":
            continue
        for executor_key in node.executor_keys:
            if executor_key not in per_spec_limits:
                missing_keys.add(executor_key)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"max_workers_total mapping is missing executor spec limits for: {missing}"
        )


def _run_dir(run_root: Path | None) -> Path:
    base = run_root or (FURU_CONFIG.base_root / "runs")
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:6]
    run_dir = base / f"{stamp}-{token}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _spec_with_pool_worker_logs(spec: SlurmSpec, worker_dir: Path) -> SlurmSpec:
    merged_extra: dict[str, SlurmSpecExtraValue] = {}
    if spec.extra is not None:
        merged_extra.update(spec.extra)

    raw_additional = merged_extra.get("slurm_additional_parameters")
    additional_params: dict[str, SlurmSpecExtraValue]
    if raw_additional is None:
        additional_params = {}
    else:
        if not isinstance(raw_additional, Mapping):
            raise TypeError(
                "slurm_additional_parameters must be a mapping when provided."
            )
        additional_params = dict(
            cast(Mapping[str, SlurmSpecExtraValue], raw_additional)
        )

    additional_params["output"] = str(worker_dir / "stdout.log")
    additional_params["error"] = str(worker_dir / "stderr.log")
    merged_extra["slurm_additional_parameters"] = additional_params

    return SlurmSpec(
        partition=spec.partition,
        gpus=spec.gpus,
        cpus=spec.cpus,
        mem_gb=spec.mem_gb,
        time_min=spec.time_min,
        extra=merged_extra,
    )


def _queue_root(run_dir: Path) -> Path:
    return run_dir / "queue"


def _todo_dir(run_dir: Path, spec_key: str) -> Path:
    return _queue_root(run_dir) / "todo" / spec_key


def _running_dir(run_dir: Path, spec_key: str) -> Path:
    return _queue_root(run_dir) / "running" / spec_key


def _done_dir(run_dir: Path) -> Path:
    return _queue_root(run_dir) / "done"


def _failed_dir(run_dir: Path) -> Path:
    return _queue_root(run_dir) / "failed"


@dataclass
class _FailedQueueEntry:
    path: Path
    payload: _TaskPayload | None
    parse_error: str | None


def _read_failed_entry(path: Path) -> _FailedQueueEntry:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return _FailedQueueEntry(path=path, payload=None, parse_error=str(exc))
    if not isinstance(payload, dict):
        return _FailedQueueEntry(
            path=path,
            payload=None,
            parse_error="Failed payload is not a JSON object",
        )
    return _FailedQueueEntry(
        path=path, payload=cast(_TaskPayload, payload), parse_error=None
    )


def _scan_failed_tasks(run_dir: Path) -> list[_FailedQueueEntry]:
    failed_root = _failed_dir(run_dir)
    if not failed_root.exists():
        return []
    failed_files = sorted(failed_root.rglob("*.json"))
    if not failed_files:
        return []
    return [_read_failed_entry(path) for path in failed_files]


def _worker_id_from_path(task_path: Path) -> str | None:
    parts = task_path.parts
    try:
        running_index = parts.index("running")
    except ValueError:
        return None
    if len(parts) <= running_index + 2:
        return None
    return parts[running_index + 2]


def _format_failed_entry(entry: _FailedQueueEntry, *, reason: str) -> str:
    payload = entry.payload or {}
    task_hash = payload.get("hash") or entry.path.stem
    spec_key = payload.get("spec_key") or "unknown"
    worker_id = payload.get("worker_id") or "unknown"
    attempt = payload.get("attempt")
    attempt_str = str(attempt) if isinstance(attempt, int) else "unknown"
    failure_kind = payload.get("failure_kind") or "unknown"
    error = payload.get("error") or "unknown"
    lines = [
        "run_slurm_pool stopped: failed task entry detected in queue/failed.",
        f"Reason: {reason}",
        f"path: {entry.path}",
        f"hash: {task_hash}",
        f"spec_key: {spec_key}",
        f"worker_id: {worker_id}",
        f"attempt: {attempt_str}",
        f"failure_kind: {failure_kind}",
        f"error: {error}",
    ]
    if entry.parse_error is not None:
        lines.append(f"parse_error: {entry.parse_error}")
    return "\n".join(lines)


def _requeue_failed_task(
    run_dir: Path,
    entry: _FailedQueueEntry,
    payload: _TaskPayload,
    *,
    next_attempt: int,
) -> None:
    spec_key = payload.get("spec_key")
    obj_payload = payload.get("obj")
    if not isinstance(spec_key, str):
        raise RuntimeError(
            _format_failed_entry(entry, reason="Failed entry missing spec_key")
        )
    if obj_payload is None:
        raise RuntimeError(
            _format_failed_entry(entry, reason="Failed entry missing obj payload")
        )
    updated_payload = cast(_TaskPayload, dict(payload))
    for stale_field in (
        "error",
        "failure_kind",
        "traceback",
        "failed_at",
        "claimed_at",
        "worker_id",
    ):
        updated_payload.pop(stale_field, None)
    updated_payload["attempt"] = next_attempt
    updated_payload["spec_key"] = spec_key
    updated_payload["executor_keys"] = list(
        _executor_keys_from_payload(
            updated_payload,
            fallback_spec_key=spec_key,
        )
    )
    updated_payload["obj"] = obj_payload
    task_path = _todo_dir(run_dir, spec_key) / entry.path.name
    task_path.parent.mkdir(parents=True, exist_ok=True)
    if task_path.exists():
        raise RuntimeError(
            _format_failed_entry(
                entry,
                reason=f"Retry conflict: todo entry already exists at {task_path}",
            )
        )
    _atomic_write_json(task_path, updated_payload)
    entry.path.unlink(missing_ok=True)


def _handle_failed_tasks(
    run_dir: Path,
    entries: list[_FailedQueueEntry],
    *,
    retry_failed: bool,
    max_compute_retries: int,
) -> int:
    requeued = 0
    for entry in entries:
        if entry.parse_error is not None:
            raise RuntimeError(
                _format_failed_entry(entry, reason="Invalid failed task payload")
            )
        if entry.payload is None:
            raise RuntimeError(
                _format_failed_entry(entry, reason="Missing failed task payload")
            )
        payload = entry.payload
        failure_kind = payload.get("failure_kind")
        if failure_kind != "compute":
            raise RuntimeError(
                _format_failed_entry(entry, reason="Protocol failure in failed queue")
            )
        attempt = payload.get("attempt")
        if not isinstance(attempt, int):
            raise RuntimeError(
                _format_failed_entry(entry, reason="Failed entry missing attempt count")
            )
        if retry_failed and attempt <= max_compute_retries:
            _requeue_failed_task(
                run_dir,
                entry,
                payload,
                next_attempt=attempt + 1,
            )
            requeued += 1
            continue
        if not retry_failed:
            reason = "Compute failure with retry_failed disabled"
        else:
            retries_used = max(attempt - 1, 0)
            reason = (
                "Compute failure exhausted retries "
                f"({retries_used}/{max_compute_retries})"
            )
        raise RuntimeError(_format_failed_entry(entry, reason=reason))
    return requeued


def _done_hashes(run_dir: Path) -> set[str]:
    done_dir = _done_dir(run_dir)
    if not done_dir.exists():
        return set()
    return {path.stem for path in done_dir.iterdir() if path.is_file()}


def _task_filename(digest: str) -> str:
    return f"{digest}.json"


def _atomic_write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Write a JSON payload atomically.

    Queue entries are consumed by other processes (workers/controllers). Using a
    temp file + atomic rename avoids readers observing partially-written JSON on
    shared/network filesystems.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def _ensure_queue_layout(
    run_dir: Path, specs: Mapping[str, SlurmSpec] | None = None
) -> None:
    queue_root = _queue_root(run_dir)
    (queue_root / "todo").mkdir(parents=True, exist_ok=True)
    (queue_root / "running").mkdir(parents=True, exist_ok=True)
    _done_dir(run_dir).mkdir(parents=True, exist_ok=True)
    _failed_dir(run_dir).mkdir(parents=True, exist_ok=True)
    if specs is None:
        return
    for spec_key in specs:
        _todo_dir(run_dir, spec_key).mkdir(parents=True, exist_ok=True)
        _running_dir(run_dir, spec_key).mkdir(parents=True, exist_ok=True)


def _executor_keys_from_payload(
    payload: Mapping[str, JsonValue],
    *,
    fallback_spec_key: str | None = None,
) -> tuple[str, ...]:
    keys: list[str] = []
    raw_keys = payload.get("executor_keys")
    if isinstance(raw_keys, list):
        for key in raw_keys:
            if not isinstance(key, str):
                continue
            if not key:
                continue
            if key in keys:
                continue
            keys.append(key)
    if keys:
        return tuple(keys)

    raw_spec_key = payload.get("spec_key")
    if isinstance(raw_spec_key, str) and raw_spec_key:
        return (raw_spec_key,)
    if fallback_spec_key is not None:
        return (fallback_spec_key,)
    return tuple()


def _task_known(run_dir: Path, digest: str) -> bool:
    filename = _task_filename(digest)
    todo_root = _queue_root(run_dir) / "todo"
    if todo_root.exists():
        for path in todo_root.glob(f"*/{filename}"):
            if path.exists():
                return True
    if (_done_dir(run_dir) / filename).exists():
        return True
    if (_failed_dir(run_dir) / filename).exists():
        return True
    running_root = _queue_root(run_dir) / "running"
    if running_root.exists():
        for path in running_root.glob(f"*/*/{filename}"):
            if path.exists():
                return True
    return False


def _enqueue_task(
    run_dir: Path,
    node_hash: str,
    spec_key: str,
    executor_keys: tuple[str, ...],
    obj: Furu,
) -> bool:
    if _task_known(run_dir, node_hash):
        return False

    normalized_executor_keys: list[str] = []
    for key in executor_keys:
        if key in normalized_executor_keys:
            continue
        normalized_executor_keys.append(key)
    if spec_key not in normalized_executor_keys:
        normalized_executor_keys.insert(0, spec_key)

    payload: _TaskPayload = {
        "hash": node_hash,
        "spec_key": spec_key,
        "executor_keys": normalized_executor_keys,
        "obj": obj.to_dict(),
        "attempt": 1,
    }
    path = _todo_dir(run_dir, spec_key) / _task_filename(node_hash)
    _atomic_write_json(path, payload)
    return True


def _claim_task(run_dir: Path, worker_spec_key: str, worker_id: str) -> Path | None:
    todo_root = _queue_root(run_dir) / "todo"
    if not todo_root.exists():
        return None
    running_root = _running_dir(run_dir, worker_spec_key) / worker_id
    running_root.mkdir(parents=True, exist_ok=True)

    for path in sorted(todo_root.glob("*/*.json")):
        if not path.is_file():
            continue
        fallback_spec_key = path.parent.name
        should_claim = False
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            should_claim = True
        else:
            if isinstance(raw, dict):
                payload = cast(_TaskPayload, raw)
                task_executor_keys = _executor_keys_from_payload(
                    payload,
                    fallback_spec_key=fallback_spec_key,
                )
                should_claim = worker_spec_key in task_executor_keys
            else:
                should_claim = True
        if not should_claim:
            continue
        target = running_root / path.name
        try:
            path.replace(target)
        except FileNotFoundError:
            continue
        now = time.time()
        with contextlib.suppress(OSError):
            os.utime(target, (now, now))

        # Best-effort: persist an explicit claim timestamp. This makes missing-heartbeat
        # grace robust on filesystems with coarse mtimes or unexpected mtime behavior.
        try:
            raw = json.loads(target.read_text())
            if isinstance(raw, dict):
                payload = cast(_TaskPayload, raw)
                payload["claimed_at"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                payload["worker_id"] = worker_id
                if not isinstance(payload.get("spec_key"), str):
                    payload["spec_key"] = fallback_spec_key
                payload["executor_keys"] = list(
                    _executor_keys_from_payload(
                        payload,
                        fallback_spec_key=fallback_spec_key,
                    )
                )
                tmp = target.with_suffix(f".tmp-{uuid.uuid4().hex}")
                tmp.write_text(json.dumps(payload, indent=2))
                tmp.replace(target)
        except Exception as exc:
            logger = get_logger()
            logger.warning(
                "pool claim: failed to stamp claimed_at/worker_id for %s: %s",
                target,
                exc,
            )
        return target
    return None


def _heartbeat_path(task_path: Path) -> Path:
    return task_path.with_suffix(".hb")


def _touch_heartbeat(path: Path) -> None:
    now = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        if path.exists():
            os.utime(path, (now, now))
            return
        path.touch()


def _heartbeat_loop(
    path: Path, interval_sec: float, stop_event: threading.Event
) -> None:
    while not stop_event.wait(interval_sec):
        _touch_heartbeat(path)


def _mark_done(run_dir: Path, task_path: Path) -> None:
    target = _done_dir(run_dir) / task_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    hb_path = _heartbeat_path(task_path)
    try:
        task_path.replace(target)
    except FileNotFoundError:
        hb_path.unlink(missing_ok=True)
        return
    hb_path.unlink(missing_ok=True)


def _mark_failed(
    run_dir: Path,
    task_path: Path,
    message: str,
    *,
    failure_kind: FailureKind,
) -> None:
    payload: _TaskPayload = {
        "hash": task_path.stem,
        "error": message,
        "failure_kind": failure_kind,
        "attempt": 1,
    }
    try:
        raw_payload = json.loads(task_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        raw_payload = None
    if isinstance(raw_payload, dict):
        payload.update(cast(_TaskPayload, raw_payload))
        payload["error"] = message
        payload["failure_kind"] = failure_kind
        if not isinstance(payload.get("attempt"), int):
            payload["attempt"] = 1
    if "worker_id" not in payload:
        worker_id = _worker_id_from_path(task_path)
        if worker_id is not None:
            payload["worker_id"] = worker_id
    target = _failed_dir(run_dir) / task_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target, payload)
    task_path.unlink(missing_ok=True)
    _heartbeat_path(task_path).unlink(missing_ok=True)


def pool_worker_main(
    run_dir: Path,
    spec_key: str,
    idle_timeout_sec: float,
    poll_interval_sec: float,
    worker_id: str | None = None,
) -> None:
    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    last_task_time = time.time()

    while True:
        task_path = _claim_task(run_dir, spec_key, worker_id)
        if task_path is None:
            if time.time() - last_task_time > idle_timeout_sec:
                return
            time.sleep(poll_interval_sec)
            continue

        last_task_time = time.time()
        try:
            payload = json.loads(task_path.read_text())
        except json.JSONDecodeError as exc:
            _mark_failed(
                run_dir,
                task_path,
                f"Invalid task payload JSON: {exc}",
                failure_kind=classify_pool_exception(exc, phase="payload"),
            )
            raise
        obj_payload = payload.get("obj") if isinstance(payload, dict) else None
        if obj_payload is None:
            _mark_failed(
                run_dir,
                task_path,
                "Missing task payload",
                failure_kind="protocol",
            )
            raise RuntimeError("Missing task payload")

        try:
            obj = Furu.from_dict(obj_payload)
        except Exception as exc:
            _mark_failed(
                run_dir,
                task_path,
                f"Invalid task payload: {exc}",
                failure_kind=classify_pool_exception(exc, phase="payload"),
            )
            raise
        if not isinstance(obj, Furu):
            message = f"Invalid task payload: expected Furu, got {type(obj).__name__}"
            _mark_failed(run_dir, task_path, message, failure_kind="protocol")
            raise RuntimeError(message)
        task_executor_keys: list[str] = []
        for task_spec in resolve_executor_specs(obj):
            key = slurm_spec_key(task_spec)
            if key in task_executor_keys:
                continue
            task_executor_keys.append(key)
        if spec_key not in task_executor_keys:
            message = (
                "Spec mismatch: task allows "
                f"{tuple(task_executor_keys)!r} on worker {spec_key}"
            )
            _mark_failed(run_dir, task_path, message, failure_kind="protocol")
            raise RuntimeError(message)

        hb_path = _heartbeat_path(task_path)
        _touch_heartbeat(hb_path)
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(hb_path, max(0.5, poll_interval_sec), heartbeat_stop),
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            obj._worker_entry(
                allow_failed=FURU_CONFIG.retry_failed,
                worker_spec_key=spec_key,
            )
        except Exception as exc:
            heartbeat_stop.set()
            heartbeat_thread.join()
            state = obj.get_state()
            failure_kind = classify_pool_exception(
                exc,
                phase="worker",
                state=state,
            )
            _mark_failed(run_dir, task_path, str(exc), failure_kind=failure_kind)
            raise
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join()

        state = obj.get_state()
        if isinstance(state.result, _StateResultSuccess):
            _mark_done(run_dir, task_path)
            continue

        if isinstance(state.result, _StateResultFailed):
            message = "Task failed; furu state is failed"
            failure_kind: FailureKind = "compute"
        else:
            message = (
                f"Task did not complete successfully (state={state.result.status})"
            )
            failure_kind = "protocol"
        _mark_failed(run_dir, task_path, message, failure_kind=failure_kind)
        raise RuntimeError(message)


def _backlog(run_dir: Path, spec_key: str) -> int:
    todo_root = _queue_root(run_dir) / "todo"
    if not todo_root.exists():
        return 0

    backlog = 0
    for path in todo_root.glob("*/*.json"):
        if not path.is_file():
            continue
        fallback_spec_key = path.parent.name
        try:
            raw_payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            backlog += 1
            continue
        if not isinstance(raw_payload, dict):
            backlog += 1
            continue
        payload = cast(_TaskPayload, raw_payload)
        executor_keys = _executor_keys_from_payload(
            payload,
            fallback_spec_key=fallback_spec_key,
        )
        if spec_key in executor_keys:
            backlog += 1
    return backlog


def _requeue_stale_running(
    run_dir: Path,
    *,
    stale_sec: float,
    heartbeat_grace_sec: float,
    max_compute_retries: int,
) -> int:
    running_root = _queue_root(run_dir) / "running"
    if not running_root.exists():
        return 0

    now = time.time()
    moved = 0
    for path in sorted(running_root.rglob("*.json")):
        hb_path = _heartbeat_path(path)
        try:
            hb_mtime = hb_path.stat().st_mtime
        except FileNotFoundError:
            hb_mtime = None
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if hb_mtime is None:
            if now - mtime <= heartbeat_grace_sec:
                continue
            try:
                raw_payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                raw_payload = None
            if not isinstance(raw_payload, dict):
                message = (
                    "Missing heartbeat file for running task beyond grace period; "
                    "invalid payload."
                )
                _mark_failed(run_dir, path, message, failure_kind="protocol")
                continue
            payload = cast(_TaskPayload, raw_payload)
            claimed_at = payload.get("claimed_at")
            if isinstance(claimed_at, str):
                normalized = claimed_at.replace("Z", "+00:00")
                try:
                    claimed_dt = datetime.fromisoformat(normalized)
                except ValueError:
                    logger = get_logger()
                    logger.warning(
                        "pool controller: invalid claimed_at=%r for %s; falling back to mtime",
                        claimed_at,
                        path,
                    )
                else:
                    if claimed_dt.tzinfo is None:
                        claimed_dt = claimed_dt.replace(tzinfo=timezone.utc)
                    if now - claimed_dt.timestamp() <= heartbeat_grace_sec:
                        continue
            requeues = payload.get("missing_heartbeat_requeues")
            requeues_count = requeues if isinstance(requeues, int) else 0
            if requeues_count < MISSING_HEARTBEAT_REQUEUE_LIMIT:
                if len(path.parents) < 3:
                    continue
                fallback_spec_key = path.parent.parent.name
                raw_spec_key = payload.get("spec_key")
                spec_key = (
                    raw_spec_key
                    if isinstance(raw_spec_key, str) and raw_spec_key
                    else fallback_spec_key
                )
                target = _todo_dir(run_dir, spec_key) / path.name
                if target.exists():
                    logger = get_logger()
                    logger.warning(
                        "run_slurm_pool: missing-heartbeat requeue found existing todo %s; cleaning up stale running entry %s",
                        target,
                        path,
                    )
                    path.unlink(missing_ok=True)
                    hb_path.unlink(missing_ok=True)
                    continue
                updated_payload = dict(payload)
                updated_payload["spec_key"] = spec_key
                updated_payload["executor_keys"] = list(
                    _executor_keys_from_payload(
                        payload,
                        fallback_spec_key=spec_key,
                    )
                )
                updated_payload["missing_heartbeat_requeues"] = requeues_count + 1
                updated_payload.pop("claimed_at", None)
                updated_payload.pop("worker_id", None)
                _atomic_write_json(target, updated_payload)
                path.unlink(missing_ok=True)
                hb_path.unlink(missing_ok=True)
                moved += 1
                continue
            message = (
                "Missing heartbeat file for running task beyond grace period; "
                "missing-heartbeat requeues exhausted."
            )
            _mark_failed(run_dir, path, message, failure_kind="protocol")
            continue
        if now - hb_mtime <= stale_sec:
            continue
        if len(path.parents) < 3:
            continue
        try:
            raw_payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            raw_payload = None
        if not isinstance(raw_payload, dict):
            message = "Stale heartbeat beyond threshold; invalid payload."
            _mark_failed(run_dir, path, message, failure_kind="protocol")
            raise RuntimeError(message)
        payload = cast(_TaskPayload, raw_payload)
        attempt = payload.get("attempt")
        if not isinstance(attempt, int):
            message = "Stale heartbeat beyond threshold; missing attempt count."
            _mark_failed(run_dir, path, message, failure_kind="protocol")
            raise RuntimeError(message)
        if attempt > max_compute_retries:
            retries_used = max(attempt - 1, 0)
            message = (
                "Stale heartbeat exhausted retries "
                f"({retries_used}/{max_compute_retries})."
            )
            _mark_failed(run_dir, path, message, failure_kind="protocol")
            raise RuntimeError(message)
        fallback_spec_key = path.parent.parent.name
        raw_spec_key = payload.get("spec_key")
        spec_key = (
            raw_spec_key
            if isinstance(raw_spec_key, str) and raw_spec_key
            else fallback_spec_key
        )
        target = _todo_dir(run_dir, spec_key) / path.name
        if target.exists():
            logger = get_logger()
            logger.warning(
                "run_slurm_pool: stale-heartbeat requeue found existing todo %s; cleaning up stale running entry %s",
                target,
                path,
            )
            path.unlink(missing_ok=True)
            hb_path.unlink(missing_ok=True)
            continue
        requeues = payload.get("stale_heartbeat_requeues")
        requeues_count = requeues if isinstance(requeues, int) else 0
        updated_payload = dict(payload)
        updated_payload["spec_key"] = spec_key
        updated_payload["executor_keys"] = list(
            _executor_keys_from_payload(
                payload,
                fallback_spec_key=spec_key,
            )
        )
        updated_payload["attempt"] = attempt + 1
        updated_payload["stale_heartbeat_requeues"] = requeues_count + 1
        updated_payload.pop("claimed_at", None)
        updated_payload.pop("worker_id", None)
        _atomic_write_json(target, updated_payload)
        path.unlink(missing_ok=True)
        hb_path.unlink(missing_ok=True)
        moved += 1
    return moved


@dataclass(frozen=True)
class _InProgressOwnership:
    self_hashes: set[str]
    external_hashes: set[str]


def _in_progress_ownership(
    plan: DependencyPlan,
    *,
    queued_hashes: set[str],
) -> _InProgressOwnership:
    self_hashes = {
        digest
        for digest in queued_hashes
        if digest in plan.nodes and plan.nodes[digest].status == "IN_PROGRESS"
    }
    external_hashes = {
        digest
        for digest, node in plan.nodes.items()
        if node.status == "IN_PROGRESS" and digest not in self_hashes
    }
    return _InProgressOwnership(
        self_hashes=self_hashes,
        external_hashes=external_hashes,
    )


def _next_refresh_deadline(now: float, interval_sec: float) -> float:
    if interval_sec <= 0:
        return now
    return now + interval_sec


def _resolve_interval(
    *,
    interval_sec: float | None,
    fallback_sec: float,
    name: str,
) -> float:
    effective = fallback_sec if interval_sec is None else interval_sec
    if effective < 0:
        raise ValueError(f"{name} must be >= 0")
    return effective


def _refresh_ready_batch(
    plan: DependencyPlan,
    *,
    queued_hashes: set[str],
    done_hashes: set[str],
) -> list[str]:
    return [
        digest
        for digest in ready_todo(plan)
        if digest not in queued_hashes and digest not in done_hashes
    ]


def run_slurm_pool(
    roots: list[Furu],
    *,
    max_workers_total: WorkerLimitConfig = 50,
    window_size: str | int = "bfs",
    idle_timeout_sec: float = 60.0,
    poll_interval_sec: float = 2.0,
    plan_refresh_interval_sec: float = 60.0,
    worker_health_check_interval_sec: float = 15.0,
    done_scan_interval_sec: float | None = None,
    failed_scan_interval_sec: float | None = None,
    stale_scan_interval_sec: float | None = None,
    ready_refresh_interval_sec: float | None = None,
    stale_running_sec: float = 900.0,
    heartbeat_grace_sec: float = 30.0,
    submitit_root: Path | None = None,
    run_root: Path | None = None,
) -> SlurmPoolRun:
    worker_limits = _normalize_worker_limits(max_workers_total)
    if plan_refresh_interval_sec < 0:
        raise ValueError("plan_refresh_interval_sec must be >= 0")
    if worker_health_check_interval_sec < 0:
        raise ValueError("worker_health_check_interval_sec must be >= 0")

    done_scan_interval = _resolve_interval(
        interval_sec=done_scan_interval_sec,
        fallback_sec=poll_interval_sec,
        name="done_scan_interval_sec",
    )
    failed_scan_interval = _resolve_interval(
        interval_sec=failed_scan_interval_sec,
        fallback_sec=poll_interval_sec,
        name="failed_scan_interval_sec",
    )
    stale_scan_interval = _resolve_interval(
        interval_sec=stale_scan_interval_sec,
        fallback_sec=poll_interval_sec,
        name="stale_scan_interval_sec",
    )
    ready_refresh_interval = _resolve_interval(
        interval_sec=ready_refresh_interval_sec,
        fallback_sec=poll_interval_sec,
        name="ready_refresh_interval_sec",
    )

    run_dir = _run_dir(run_root)
    run_id = run_dir.name
    submitit_root_effective = submitit_root_dir(submitit_root)
    _ensure_queue_layout(run_dir)

    window = _normalize_window_size(window_size, len(roots))
    active_indices = list(range(min(window, len(roots))))
    next_index = len(active_indices)
    jobs_by_spec: dict[str, list[SubmititJob]] = {}
    specs_by_key: dict[str, SlurmSpec] = {}
    job_adapter = SubmititAdapter(executor=None)
    plan: DependencyPlan | None = None
    plan_signature: tuple[int, ...] = tuple()
    done_hashes_cache: set[str] = _done_hashes(run_dir)
    queued_hashes: set[str] = set()
    next_plan_refresh_at = 0.0
    next_worker_health_check_at = 0.0
    next_done_scan_at = 0.0
    next_failed_scan_at = 0.0
    next_stale_scan_at = 0.0
    next_ready_refresh_at = 0.0
    ready_dirty = True

    while True:
        active_roots = [roots[index] for index in active_indices]
        active_signature = tuple(active_indices)

        now = time.time()
        if done_scan_interval <= 0 or now >= next_done_scan_at:
            latest_done_hashes = _done_hashes(run_dir)
            new_done_hashes = latest_done_hashes - done_hashes_cache
            if new_done_hashes:
                done_hashes_cache.update(new_done_hashes)
                queued_hashes.difference_update(new_done_hashes)
                ready_dirty = True
                if plan is not None:
                    for digest in new_done_hashes:
                        node = plan.nodes.get(digest)
                        if node is not None:
                            node.status = "DONE"
            if done_scan_interval > 0:
                next_done_scan_at = _next_refresh_deadline(now, done_scan_interval)

        needs_plan_refresh = (
            plan is None
            or plan_signature != active_signature
            or now >= next_plan_refresh_at
        )
        if needs_plan_refresh:
            plan = build_plan(active_roots, completed_hashes=done_hashes_cache)
            plan_signature = active_signature
            done_hashes_cache.update(
                digest for digest, node in plan.nodes.items() if node.status == "DONE"
            )
            queued_hashes.intersection_update(plan.nodes)
            ready_dirty = True
            next_plan_refresh_at = _next_refresh_deadline(
                now,
                plan_refresh_interval_sec,
            )
            _validate_worker_limits_for_plan(
                plan,
                per_spec_limits=worker_limits.per_spec_limits,
            )
        if plan is None:
            raise RuntimeError("internal error: slurm pool plan not initialized")

        now = time.time()
        if failed_scan_interval <= 0 or now >= next_failed_scan_at:
            failed_entries = _scan_failed_tasks(run_dir)
            if failed_entries:
                requeued_failed = _handle_failed_tasks(
                    run_dir,
                    failed_entries,
                    retry_failed=FURU_CONFIG.retry_failed,
                    max_compute_retries=FURU_CONFIG.max_compute_retries,
                )
                if requeued_failed > 0:
                    ready_dirty = True
                    next_plan_refresh_at = 0.0
            if failed_scan_interval > 0:
                next_failed_scan_at = _next_refresh_deadline(now, failed_scan_interval)

        now = time.time()
        if stale_scan_interval <= 0 or now >= next_stale_scan_at:
            requeued_running = _requeue_stale_running(
                run_dir,
                stale_sec=stale_running_sec,
                heartbeat_grace_sec=heartbeat_grace_sec,
                max_compute_retries=FURU_CONFIG.max_compute_retries,
            )
            if requeued_running > 0:
                ready_dirty = True
                next_plan_refresh_at = 0.0
            if stale_scan_interval > 0:
                next_stale_scan_at = _next_refresh_deadline(now, stale_scan_interval)

        if not FURU_CONFIG.retry_failed:
            failed = [node for node in plan.nodes.values() if node.status == "FAILED"]
            if failed:
                names = ", ".join(
                    f"{node.obj.__class__.__name__}({node.obj.furu_hash})"
                    for node in failed
                )
                raise RuntimeError(
                    f"Cannot run slurm pool with failed dependencies: {names}"
                )

        for node in plan.nodes.values():
            if node.status != "TODO":
                continue
            for executor_key, executor_spec in zip(
                node.executor_keys,
                node.executors,
                strict=True,
            ):
                specs_by_key[executor_key] = executor_spec
                jobs_by_spec.setdefault(executor_key, [])

        ready: list[str] = []
        now = time.time()
        if ready_dirty and (
            ready_refresh_interval <= 0 or now >= next_ready_refresh_at
        ):
            ready = _refresh_ready_batch(
                plan,
                queued_hashes=queued_hashes,
                done_hashes=done_hashes_cache,
            )
            ready_dirty = False
            if ready_refresh_interval > 0:
                next_ready_refresh_at = _next_refresh_deadline(
                    now, ready_refresh_interval
                )

        for digest in ready:
            node = plan.nodes.get(digest)
            if node is None:
                continue
            for executor_key, executor_spec in zip(
                node.executor_keys,
                node.executors,
                strict=True,
            ):
                specs_by_key[executor_key] = executor_spec
                jobs_by_spec.setdefault(executor_key, [])
            if _enqueue_task(
                run_dir,
                digest,
                node.executor_key,
                node.executor_keys,
                node.obj,
            ):
                queued_hashes.add(digest)
                continue
            ready_dirty = True

        now = time.time()
        if now >= next_worker_health_check_at:
            for spec_key, jobs in list(jobs_by_spec.items()):
                jobs_by_spec[spec_key] = [
                    job for job in jobs if not job_adapter.is_done(job)
                ]
            next_worker_health_check_at = _next_refresh_deadline(
                now,
                worker_health_check_interval_sec,
            )

        total_workers = sum(len(jobs) for jobs in jobs_by_spec.values())
        known_spec_keys = sorted({*specs_by_key, *jobs_by_spec})
        workers_by_spec = {
            spec_key: len(jobs_by_spec.get(spec_key, []))
            for spec_key in known_spec_keys
        }
        backlog_by_spec = {
            spec_key: _backlog(run_dir, spec_key) for spec_key in known_spec_keys
        }

        def has_spec_capacity(spec_key: str) -> bool:
            if worker_limits.per_spec_limits is None:
                return True
            limit = worker_limits.per_spec_limits.get(spec_key)
            if limit is None:
                return False
            return workers_by_spec.get(spec_key, 0) < limit

        while total_workers < worker_limits.total_limit:
            candidate_spec_keys = [
                spec_key
                for spec_key, count in backlog_by_spec.items()
                if count > 0 and has_spec_capacity(spec_key)
            ]
            if not candidate_spec_keys:
                break
            spec_key = max(candidate_spec_keys, key=lambda key: backlog_by_spec[key])
            latest_spec_backlog = _backlog(run_dir, spec_key)
            backlog_by_spec[spec_key] = latest_spec_backlog
            if latest_spec_backlog <= 0 or not has_spec_capacity(spec_key):
                continue
            spec = specs_by_key.get(spec_key)
            if spec is None:
                raise RuntimeError(
                    f"Missing executor spec for queued tasks: spec_key={spec_key!r}."
                )
            worker_id = f"worker-{uuid.uuid4().hex[:12]}"
            worker_dir = _running_dir(run_dir, spec_key) / worker_id
            worker_dir.mkdir(parents=True, exist_ok=True)
            worker_spec = _spec_with_pool_worker_logs(spec, worker_dir)
            executor = make_executor_for_spec(
                spec_key,
                worker_spec,
                kind="workers",
                submitit_root=submitit_root_effective,
                run_id=run_id,
            )
            adapter = SubmititAdapter(executor)
            job = adapter.submit(
                lambda: pool_worker_main(
                    run_dir,
                    spec_key,
                    idle_timeout_sec=idle_timeout_sec,
                    poll_interval_sec=poll_interval_sec,
                    worker_id=worker_id,
                )
            )
            jobs_by_spec[spec_key].append(job)
            total_workers += 1
            workers_by_spec[spec_key] = workers_by_spec.get(spec_key, 0) + 1
            backlog_by_spec[spec_key] = max(0, latest_spec_backlog - 1)

        finished_indices = [
            index
            for index in active_indices
            if plan.nodes.get(roots[index].furu_hash) is not None
            and plan.nodes[roots[index].furu_hash].status == "DONE"
        ]
        for index in finished_indices:
            active_indices.remove(index)

        while len(active_indices) < window and next_index < len(roots):
            active_indices.append(next_index)
            next_index += 1

        ownership = _in_progress_ownership(plan, queued_hashes=queued_hashes)

        ready_for_stall = ready
        if not ready_for_stall and ready_dirty:
            ready_for_stall = _refresh_ready_batch(
                plan,
                queued_hashes=queued_hashes,
                done_hashes=done_hashes_cache,
            )
            ready_dirty = False
            now = time.time()
            if ready_refresh_interval > 0:
                next_ready_refresh_at = _next_refresh_deadline(
                    now, ready_refresh_interval
                )

        if not active_indices and next_index >= len(roots):
            return SlurmPoolRun(
                run_dir=run_dir,
                submitit_root=submitit_root_effective,
                plan=plan,
            )
        if (
            not ready_for_stall
            and total_workers == 0
            and not any(count > 0 for count in backlog_by_spec.values())
            and not (ownership.self_hashes or ownership.external_hashes)
        ):
            todo_nodes = [node for node in plan.nodes.values() if node.status == "TODO"]
            if todo_nodes:
                sample = ", ".join(
                    f"{node.obj.__class__.__name__}({node.obj.furu_hash})"
                    for node in todo_nodes[:3]
                )
                raise RuntimeError(
                    "run_slurm_pool stalled with no progress; "
                    f"remaining TODO nodes: {sample}"
                )

        if ownership.self_hashes or ownership.external_hashes:
            stale_detected = reconcile_or_timeout_in_progress(
                plan,
                stale_timeout_sec=FURU_CONFIG.stale_timeout,
            )
            if stale_detected:
                next_plan_refresh_at = 0.0
                ready_dirty = True
                continue

        now = time.time()
        sleep_duration = poll_interval_sec
        if done_scan_interval > 0:
            sleep_duration = min(
                sleep_duration,
                max(0.0, next_done_scan_at - now),
            )
        if failed_scan_interval > 0:
            sleep_duration = min(
                sleep_duration,
                max(0.0, next_failed_scan_at - now),
            )
        if stale_scan_interval > 0:
            sleep_duration = min(
                sleep_duration,
                max(0.0, next_stale_scan_at - now),
            )
        if ready_refresh_interval > 0 and ready_dirty:
            sleep_duration = min(
                sleep_duration,
                max(0.0, next_ready_refresh_at - now),
            )
        if worker_health_check_interval_sec > 0:
            sleep_duration = min(
                sleep_duration,
                max(0.0, next_worker_health_check_at - now),
            )
        if plan_refresh_interval_sec > 0:
            sleep_duration = min(
                sleep_duration,
                max(0.0, next_plan_refresh_at - now),
            )

        time.sleep(sleep_duration)
