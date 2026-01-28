from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Literal

from ..config import FURU_CONFIG
from ..core import Furu
from ..errors import FuruValidationError
from ..runtime.logging import get_logger
from ..storage.migration import MigrationManager, MigrationRecord
from ..storage.state import (
    StateManager,
    _StateAttemptFailed,
    _StateAttemptQueued,
    _StateAttemptRunning,
    _FuruState,
    _StateResultFailed,
)

Status = Literal["DONE", "IN_PROGRESS", "TODO", "FAILED"]

_MISSING_TIMESTAMP_SEEN: dict[str, float] = {}


@dataclass
class PlanNode:
    obj: Furu
    status: Status
    spec_key: str
    deps_all: set[str]
    deps_pending: set[str]
    dependents: set[str]


@dataclass
class DependencyPlan:
    roots: list[Furu]
    nodes: dict[str, PlanNode]


@dataclass
class _PlanCache:
    migration_records: dict[Path, MigrationRecord | None]
    alias_targets: dict[Path, Path | None]
    marker_exists: dict[Path, bool]
    states: dict[Path, _FuruState]


def _marker_exists(directory: Path, cache: _PlanCache) -> bool:
    if directory in cache.marker_exists:
        return cache.marker_exists[directory]
    exists = StateManager.success_marker_exists(directory)
    cache.marker_exists[directory] = exists
    return exists


def _migration_record(directory: Path, cache: _PlanCache) -> MigrationRecord | None:
    if directory not in cache.migration_records:
        cache.migration_records[directory] = MigrationManager.read_migration(directory)
    return cache.migration_records[directory]


def _alias_target_dir(base_dir: Path, cache: _PlanCache) -> Path | None:
    if base_dir in cache.alias_targets:
        return cache.alias_targets[base_dir]
    record = _migration_record(base_dir, cache)
    if record is None or record.kind != "alias" or record.overwritten_at is not None:
        cache.alias_targets[base_dir] = None
        return None
    if _marker_exists(base_dir, cache):
        cache.alias_targets[base_dir] = None
        return None
    target_dir = MigrationManager.resolve_dir(record, target="from")
    if _marker_exists(target_dir, cache):
        cache.alias_targets[base_dir] = target_dir
        return target_dir
    cache.alias_targets[base_dir] = None
    return None


def _state_for(directory: Path, cache: _PlanCache) -> _FuruState:
    if directory not in cache.states:
        cache.states[directory] = StateManager.read_state(directory)
    return cache.states[directory]


def _validate_cached(obj: Furu, *, directory: Path) -> bool:
    try:
        return obj._validate()
    except FuruValidationError as exc:
        logger = get_logger()
        logger.warning(
            "exists %s -> false (validate invalid for %s: %s)",
            directory,
            f"{obj.__class__.__name__}({obj.furu_hash})",
            exc,
        )
        return False
    except Exception as exc:
        logger = get_logger()
        logger.exception(
            "exists %s -> false (validate crashed for %s: %s)",
            directory,
            f"{obj.__class__.__name__}({obj.furu_hash})",
            exc,
        )
        return False


def _classify(
    obj: Furu,
    completed_hashes: set[str] | None,
    cache: _PlanCache,
) -> Status:
    if completed_hashes is not None and obj.furu_hash in completed_hashes:
        return "DONE"
    base_dir = obj._base_furu_dir()
    alias_target = None
    if not obj._always_rerun():
        alias_target = _alias_target_dir(base_dir, cache)
        success_dir = alias_target or base_dir
        if _marker_exists(success_dir, cache):
            if _validate_cached(obj, directory=base_dir):
                return "DONE"

    state_dir = alias_target or base_dir
    state = _state_for(state_dir, cache)
    attempt = state.attempt
    if isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning)):
        return "IN_PROGRESS"
    if isinstance(state.result, _StateResultFailed) or isinstance(
        attempt, _StateAttemptFailed
    ):
        if FURU_CONFIG.retry_failed:
            return "TODO"
        return "FAILED"
    return "TODO"


def build_plan(
    roots: list[Furu],
    *,
    completed_hashes: set[str] | None = None,
) -> DependencyPlan:
    cache = _PlanCache(
        migration_records={},
        alias_targets={},
        marker_exists={},
        states={},
    )
    nodes: dict[str, PlanNode] = {}
    stack = list(roots)
    seen: set[str] = set()

    while stack:
        obj = stack.pop()
        digest = obj.furu_hash
        if digest in seen:
            continue
        seen.add(digest)

        status = _classify(obj, completed_hashes, cache)
        node = PlanNode(
            obj=obj,
            status=status,
            spec_key=obj._executor_spec_key(),
            deps_all=set(),
            deps_pending=set(),
            dependents=set(),
        )
        nodes[digest] = node

        if status != "TODO":
            continue

        deps = obj._get_dependencies(recursive=False)
        node.deps_all = {dep.furu_hash for dep in deps}
        for dep in deps:
            stack.append(dep)

    for digest, node in nodes.items():
        if node.status != "TODO":
            continue
        node.deps_pending = {
            dep for dep in node.deps_all if dep in nodes and nodes[dep].status != "DONE"
        }

    for digest, node in nodes.items():
        for dep in node.deps_pending:
            nodes[dep].dependents.add(digest)

    return DependencyPlan(roots=roots, nodes=nodes)


def topo_order_todo(plan: DependencyPlan) -> list[str]:
    todo = {digest for digest, node in plan.nodes.items() if node.status == "TODO"}
    indeg = {digest: 0 for digest in todo}

    for digest in todo:
        node = plan.nodes[digest]
        for dep in node.deps_pending:
            if dep in todo:
                indeg[digest] += 1

    ready = sorted([digest for digest, deg in indeg.items() if deg == 0])
    out: list[str] = []

    while ready:
        digest = ready.pop(0)
        out.append(digest)
        for dep in plan.nodes[digest].dependents:
            if dep not in todo:
                continue
            indeg[dep] -= 1
            if indeg[dep] == 0:
                ready.append(dep)
                ready.sort()

    if len(out) != len(todo):
        raise ValueError("Cycle detected in TODO dependency graph")
    return out


def ready_todo(plan: DependencyPlan) -> list[str]:
    return sorted(
        [
            digest
            for digest, node in plan.nodes.items()
            if node.status == "TODO"
            and all(plan.nodes[dep].status == "DONE" for dep in node.deps_pending)
        ]
    )


def _attempt_age_sec(
    attempt: _StateAttemptQueued | _StateAttemptRunning,
    *,
    directory: Path,
    stale_timeout_sec: float,
    digest: str,
    name: str,
) -> float | None:
    if attempt.status == "queued":
        parsed = StateManager._parse_time(attempt.started_at)
        if parsed is not None:
            _MISSING_TIMESTAMP_SEEN.pop(digest, None)
            return (StateManager._utcnow() - parsed).total_seconds()
    else:
        last_heartbeat = StateManager.last_heartbeat_mtime(directory)
        if last_heartbeat is not None:
            _MISSING_TIMESTAMP_SEEN.pop(digest, None)
            return max(0.0, time.time() - last_heartbeat)
    if stale_timeout_sec <= 0:
        return None
    now = StateManager._utcnow().timestamp()
    first_seen = _MISSING_TIMESTAMP_SEEN.get(digest)
    if first_seen is None:
        _MISSING_TIMESTAMP_SEEN[digest] = now
        logger = get_logger()
        logger.warning(
            "IN_PROGRESS attempt missing heartbeat/started timestamps for %s; "
            "deferring stale timeout check.",
            name,
        )
        return None
    return now - first_seen


def reconcile_in_progress(
    plan: DependencyPlan,
    *,
    stale_timeout_sec: float,
) -> bool:
    stale_attempts: list[
        tuple[PlanNode, _StateAttemptQueued | _StateAttemptRunning]
    ] = []
    for node in plan.nodes.values():
        if node.status != "IN_PROGRESS":
            _MISSING_TIMESTAMP_SEEN.pop(node.obj.furu_hash, None)
            continue
        state = StateManager.reconcile(node.obj._base_furu_dir())
        attempt = state.attempt
        if not isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning)):
            _MISSING_TIMESTAMP_SEEN.pop(node.obj.furu_hash, None)
            continue
        if stale_timeout_sec <= 0:
            continue
        name = f"{node.obj.__class__.__name__}({node.obj.furu_hash})"
        age = _attempt_age_sec(
            attempt,
            directory=node.obj._base_furu_dir(),
            stale_timeout_sec=stale_timeout_sec,
            digest=node.obj.furu_hash,
            name=name,
        )
        if age is None or age < stale_timeout_sec:
            continue
        stale_attempts.append((node, attempt))

    if not stale_attempts:
        return False

    names = ", ".join(
        f"{node.obj.__class__.__name__}({node.obj.furu_hash})"
        for node, _attempt in stale_attempts
    )
    if not FURU_CONFIG.retry_failed:
        raise RuntimeError(
            "Stale IN_PROGRESS dependencies detected: "
            f"{names} exceeded {stale_timeout_sec:.1f}s without heartbeat."
        )

    stale_detected = False
    for node, attempt in stale_attempts:
        stale_detected = True
        StateManager.finish_attempt_preempted(
            node.obj._base_furu_dir(),
            attempt_id=attempt.id,
            error={
                "type": "StaleHeartbeat",
                "message": (
                    f"Attempt stale after {stale_timeout_sec:.1f}s without heartbeat."
                ),
            },
            reason="stale_timeout",
        )
        _MISSING_TIMESTAMP_SEEN.pop(node.obj.furu_hash, None)
    return stale_detected
