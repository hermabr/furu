from __future__ import annotations

import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Literal

from ..config import FURU_CONFIG
from ..core import Furu
from ..errors import FuruComputeError, FuruError
from ..runtime.logging import enter_holder
from ..storage.state import (
    StateManager,
    _StateAttemptFailed,
    _StateAttemptQueued,
    _StateAttemptRunning,
    _StateResultFailed,
)
from .context import EXEC_CONTEXT, ExecContext
from .plan import DependencyPlan, PlanNode, build_plan
from .plan_utils import reconcile_or_timeout_in_progress
from .slurm_spec import SlurmSpec


RuntimeStatus = Literal[
    "READY",
    "BLOCKED",
    "IN_PROGRESS_SELF",
    "IN_PROGRESS_EXTERNAL",
    "FAILED",
    "COMPLETED",
]
LiveStatus = Literal["COMPLETED", "IN_PROGRESS_EXTERNAL", "FAILED", "TODO"]


@dataclass
class _RuntimeNode:
    obj: Furu
    executor: SlurmSpec
    executor_key: str
    deps_all: set[str]
    pending_deps: set[str]
    dependents: set[str]
    status: RuntimeStatus


@dataclass
class _SchedulerState:
    nodes: dict[str, _RuntimeNode]
    ready_queue: deque[str]
    ready_set: set[str]
    blocked_hashes: set[str]
    failed_hashes: set[str]
    in_progress_external_hashes: set[str]
    completed_hashes: set[str]
    inflight: dict[str, Future[None]]


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


def _run_node(obj: Furu, executor_key: str) -> None:
    token = EXEC_CONTEXT.set(
        ExecContext(
            mode="executor",
            spec_key=executor_key,
            backend="local",
            current_node_hash=obj.furu_hash,
        )
    )
    try:
        with enter_holder(obj):
            obj.get(force=True)
    finally:
        EXEC_CONTEXT.reset(token)


def _runtime_status_from_plan(
    status: Literal["DONE", "IN_PROGRESS", "TODO", "FAILED"],
    *,
    pending_deps: set[str],
) -> RuntimeStatus:
    if status == "DONE":
        return "COMPLETED"
    if status == "FAILED":
        return "FAILED"
    if status == "IN_PROGRESS":
        return "IN_PROGRESS_EXTERNAL"
    if pending_deps:
        return "BLOCKED"
    return "READY"


def _build_scheduler_state(plan: DependencyPlan) -> _SchedulerState:
    nodes: dict[str, _RuntimeNode] = {}
    ready_hashes: set[str] = set()
    blocked_hashes: set[str] = set()
    failed_hashes: set[str] = set()
    in_progress_external_hashes: set[str] = set()
    completed_hashes: set[str] = set()

    for digest, node in plan.nodes.items():
        pending_deps = set(node.deps_pending)
        runtime_status = _runtime_status_from_plan(
            node.status, pending_deps=pending_deps
        )
        runtime_node = _RuntimeNode(
            obj=node.obj,
            executor=node.executor,
            executor_key=node.executor_key,
            deps_all=set(node.deps_all),
            pending_deps=pending_deps,
            dependents=set(node.dependents),
            status=runtime_status,
        )
        nodes[digest] = runtime_node

        if runtime_status == "READY":
            ready_hashes.add(digest)
        elif runtime_status == "BLOCKED":
            blocked_hashes.add(digest)
        elif runtime_status == "FAILED":
            failed_hashes.add(digest)
        elif runtime_status == "IN_PROGRESS_EXTERNAL":
            in_progress_external_hashes.add(digest)
        elif runtime_status == "COMPLETED":
            completed_hashes.add(digest)

    return _SchedulerState(
        nodes=nodes,
        ready_queue=deque(sorted(ready_hashes)),
        ready_set=ready_hashes,
        blocked_hashes=blocked_hashes,
        failed_hashes=failed_hashes,
        in_progress_external_hashes=in_progress_external_hashes,
        completed_hashes=completed_hashes,
        inflight={},
    )


def _set_status(state: _SchedulerState, digest: str, status: RuntimeStatus) -> bool:
    node = state.nodes.get(digest)
    if node is None:
        return False
    previous = node.status
    if previous == "COMPLETED":
        return False
    if previous == status:
        return False

    if previous == "READY":
        state.ready_set.discard(digest)
    elif previous == "BLOCKED":
        state.blocked_hashes.discard(digest)
    elif previous == "FAILED":
        state.failed_hashes.discard(digest)
    elif previous == "IN_PROGRESS_EXTERNAL":
        state.in_progress_external_hashes.discard(digest)

    node.status = status

    if status == "READY":
        if digest not in state.ready_set:
            state.ready_set.add(digest)
            state.ready_queue.append(digest)
    elif status == "BLOCKED":
        state.blocked_hashes.add(digest)
    elif status == "FAILED":
        state.failed_hashes.add(digest)
    elif status == "IN_PROGRESS_EXTERNAL":
        state.in_progress_external_hashes.add(digest)
    elif status == "COMPLETED":
        state.completed_hashes.add(digest)

    return True


def _mark_completed(state: _SchedulerState, digest: str) -> bool:
    node = state.nodes.get(digest)
    if node is None:
        return False
    if node.status == "COMPLETED":
        return False

    changed = _set_status(state, digest, "COMPLETED")

    for dependent_digest in sorted(node.dependents):
        dependent = state.nodes.get(dependent_digest)
        if dependent is None:
            continue
        if digest not in dependent.pending_deps:
            continue
        dependent.pending_deps.discard(digest)
        if dependent.status == "BLOCKED" and not dependent.pending_deps:
            changed = _set_status(state, dependent_digest, "READY") or changed

    return changed


def _classify_live_status(
    digest: str,
    node: _RuntimeNode,
    *,
    completed_hashes: set[str],
) -> LiveStatus:
    if digest in completed_hashes:
        return "COMPLETED"

    if not node.obj._always_rerun() and node.obj._exists_quiet():
        return "COMPLETED"

    directory = node.obj._base_furu_dir()
    if not StateManager.get_internal_dir(directory).exists():
        return "TODO"
    state = StateManager.reconcile(directory)
    attempt = state.attempt
    if isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning)):
        return "IN_PROGRESS_EXTERNAL"
    if isinstance(state.result, _StateResultFailed) or isinstance(
        attempt, _StateAttemptFailed
    ):
        if FURU_CONFIG.retry_failed:
            return "TODO"
        return "FAILED"
    return "TODO"


def _apply_live_status(
    state: _SchedulerState,
    digest: str,
    live_status: LiveStatus,
) -> bool:
    node = state.nodes.get(digest)
    if node is None:
        return False

    if live_status == "COMPLETED":
        return _mark_completed(state, digest)
    if live_status == "IN_PROGRESS_EXTERNAL":
        return _set_status(state, digest, "IN_PROGRESS_EXTERNAL")
    if live_status == "FAILED":
        return _set_status(state, digest, "FAILED")

    if node.pending_deps:
        return _set_status(state, digest, "BLOCKED")
    return _set_status(state, digest, "READY")


def _recheck_hashes(state: _SchedulerState, hashes: set[str]) -> bool:
    changed = False
    for digest in sorted(hashes):
        node = state.nodes.get(digest)
        if node is None:
            continue
        if digest in state.inflight:
            continue
        if node.status == "IN_PROGRESS_SELF":
            continue
        live_status = _classify_live_status(
            digest,
            node,
            completed_hashes=state.completed_hashes,
        )
        changed = _apply_live_status(state, digest, live_status) or changed
    return changed


def _pop_ready_for_active(
    state: _SchedulerState,
    *,
    active_hashes: set[str],
) -> str | None:
    if not state.ready_queue:
        return None

    queue_len = len(state.ready_queue)
    for _ in range(queue_len):
        digest = state.ready_queue.popleft()
        if digest not in state.ready_set:
            continue
        if digest not in active_hashes:
            state.ready_queue.append(digest)
            continue
        state.ready_set.discard(digest)
        return digest
    return None


def _dispatch_ready(
    state: _SchedulerState,
    *,
    executor: ThreadPoolExecutor,
    max_workers: int,
    active_hashes: set[str],
) -> bool:
    changed = False

    while len(state.inflight) < max_workers:
        digest = _pop_ready_for_active(state, active_hashes=active_hashes)
        if digest is None:
            break
        node = state.nodes[digest]

        if node.pending_deps:
            changed = _set_status(state, digest, "BLOCKED") or changed
            continue

        live_status = _classify_live_status(
            digest,
            node,
            completed_hashes=state.completed_hashes,
        )
        if live_status != "TODO":
            changed = _apply_live_status(state, digest, live_status) or changed
            continue

        future = executor.submit(_run_node, node.obj, node.executor_key)
        state.inflight[digest] = future
        changed = _set_status(state, digest, "IN_PROGRESS_SELF") or changed

    return changed


def _handle_completed_futures(
    state: _SchedulerState,
    *,
    retry_attempts: dict[str, int],
) -> bool:
    completed_hashes = [
        digest for digest, future in state.inflight.items() if future.done()
    ]
    if not completed_hashes:
        return False

    changed = False
    for digest in completed_hashes:
        future = state.inflight.pop(digest)
        node = state.nodes.get(digest)
        if node is None:
            raise RuntimeError(
                "local executor internal error: completed hash missing runtime node"
            )
        try:
            future.result()
        except Exception as exc:
            if isinstance(exc, FuruComputeError):
                compute_error = exc
                wrapped_exc: Exception | None = None
            elif isinstance(exc, FuruError):
                raise
            else:
                state_path = StateManager.get_state_path(node.obj._base_furu_dir())
                compute_error = FuruComputeError(
                    "local executor failed for "
                    f"{node.obj.__class__.__name__}({node.obj.furu_hash})",
                    state_path,
                    original_error=exc,
                )
                wrapped_exc = exc

            if not FURU_CONFIG.retry_failed:
                if wrapped_exc is not None:
                    raise compute_error from wrapped_exc
                raise compute_error

            attempt = retry_attempts.get(digest, 0) + 1
            retry_attempts[digest] = attempt
            if attempt <= FURU_CONFIG.max_compute_retries:
                todo_status: RuntimeStatus
                if node.pending_deps:
                    todo_status = "BLOCKED"
                else:
                    todo_status = "READY"
                changed = _set_status(state, digest, todo_status) or changed
                continue

            if wrapped_exc is not None:
                raise compute_error from wrapped_exc
            raise compute_error

        retry_attempts.pop(digest, None)
        changed = _mark_completed(state, digest) or changed

    return changed


def _collect_root_scope(plan: DependencyPlan, root_hash: str) -> set[str]:
    if root_hash not in plan.nodes:
        return set()
    out: set[str] = set()
    stack = [root_hash]
    while stack:
        digest = stack.pop()
        if digest in out:
            continue
        node = plan.nodes.get(digest)
        if node is None:
            continue
        out.add(digest)
        for dep in node.deps_all:
            if dep not in out:
                stack.append(dep)
    return out


def _active_hashes(
    active_indices: list[int],
    *,
    root_scopes: dict[int, set[str]],
) -> set[str]:
    out: set[str] = set()
    for index in active_indices:
        out.update(root_scopes.get(index, set()))
    return out


def _refresh_active_indices(
    roots: list[Furu],
    *,
    state: _SchedulerState,
    active_indices: list[int],
    next_index: int,
    window: int,
) -> int:
    finished_indices = [
        index
        for index in active_indices
        if roots[index].furu_hash in state.nodes
        and state.nodes[roots[index].furu_hash].status == "COMPLETED"
    ]
    for index in finished_indices:
        active_indices.remove(index)

    while len(active_indices) < window and next_index < len(roots):
        active_indices.append(next_index)
        next_index += 1
    return next_index


def _raise_failed_active(
    state: _SchedulerState,
    *,
    active_hashes: set[str],
) -> None:
    if FURU_CONFIG.retry_failed:
        return
    failed_hashes = sorted(active_hashes & state.failed_hashes)
    if not failed_hashes:
        return
    names = ", ".join(
        f"{state.nodes[digest].obj.__class__.__name__}({digest})"
        for digest in failed_hashes
    )
    raise RuntimeError(f"Cannot run local executor with failed dependencies: {names}")


def _next_deadline(now: float, interval_sec: float) -> float:
    if interval_sec <= 0:
        return now
    return now + interval_sec


def _sample_nodes(state: _SchedulerState, hashes: set[str]) -> str:
    entries = sorted(hashes)[:3]
    return ", ".join(
        f"{state.nodes[digest].obj.__class__.__name__}({digest})" for digest in entries
    )


def _reconcile_external_stale(
    state: _SchedulerState,
    *,
    external_hashes: set[str],
    stale_timeout_sec: float,
) -> bool:
    if not external_hashes:
        return False
    nodes: dict[str, PlanNode] = {}
    roots: list[Furu] = []
    for digest in sorted(external_hashes):
        runtime_node = state.nodes.get(digest)
        if runtime_node is None:
            continue
        nodes[digest] = PlanNode(
            obj=runtime_node.obj,
            status="IN_PROGRESS",
            executor=runtime_node.executor,
            executor_key=runtime_node.executor_key,
            deps_all=set(),
            deps_pending=set(),
            dependents=set(),
        )
        roots.append(runtime_node.obj)
    if not nodes:
        return False
    plan = DependencyPlan(roots=roots, nodes=nodes)
    return reconcile_or_timeout_in_progress(
        plan,
        stale_timeout_sec=stale_timeout_sec,
    )


def run_local(
    roots: list[Furu],
    *,
    max_workers: int = 8,
    window_size: str | int = "bfs",
    poll_interval_sec: float = 0.25,
    plan_refresh_interval_sec: float = 60.0,
    underutilized_reconcile_interval_sec: float = 15.0,
    external_poll_interval_sec: float = 5.0,
) -> None:
    if not roots:
        return
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if plan_refresh_interval_sec < 0:
        raise ValueError("plan_refresh_interval_sec must be >= 0")
    if underutilized_reconcile_interval_sec < 0:
        raise ValueError("underutilized_reconcile_interval_sec must be >= 0")
    if external_poll_interval_sec < 0:
        raise ValueError("external_poll_interval_sec must be >= 0")

    window = _normalize_window_size(window_size, len(roots))
    active_indices = list(range(min(window, len(roots))))
    next_index = len(active_indices)

    plan = build_plan(roots)
    state = _build_scheduler_state(plan)
    root_scopes = {
        index: _collect_root_scope(plan, root.furu_hash)
        for index, root in enumerate(roots)
    }
    active_scope = _active_hashes(active_indices, root_scopes=root_scopes)

    retry_attempts: dict[str, int] = {}

    now = time.time()
    next_external_poll_at = _next_deadline(now, external_poll_interval_sec)
    next_underutilized_reconcile_at = _next_deadline(
        now,
        underutilized_reconcile_interval_sec,
    )
    next_full_reconcile_at = _next_deadline(now, plan_refresh_interval_sec)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            previous_active_scope = active_scope
            next_index = _refresh_active_indices(
                roots,
                state=state,
                active_indices=active_indices,
                next_index=next_index,
                window=window,
            )
            active_scope = _active_hashes(active_indices, root_scopes=root_scopes)
            newly_active_hashes = active_scope - previous_active_scope
            if newly_active_hashes and _recheck_hashes(state, newly_active_hashes):
                continue

            if not active_indices and not state.inflight and next_index >= len(roots):
                return

            _raise_failed_active(state, active_hashes=active_scope)

            if _handle_completed_futures(state, retry_attempts=retry_attempts):
                continue

            if _dispatch_ready(
                state,
                executor=executor,
                max_workers=max_workers,
                active_hashes=active_scope,
            ):
                continue

            now = time.time()
            if state.in_progress_external_hashes and (
                external_poll_interval_sec <= 0 or now >= next_external_poll_at
            ):
                changed = _recheck_hashes(
                    state,
                    set(state.in_progress_external_hashes),
                )
                if external_poll_interval_sec > 0:
                    next_external_poll_at = _next_deadline(
                        now,
                        external_poll_interval_sec,
                    )
                if changed:
                    continue

            if len(state.inflight) < max_workers and (
                underutilized_reconcile_interval_sec <= 0
                or now >= next_underutilized_reconcile_at
            ):
                recheck_hashes = (
                    set(state.ready_set)
                    | set(state.blocked_hashes)
                    | set(state.failed_hashes)
                )
                changed = _recheck_hashes(state, recheck_hashes)
                if underutilized_reconcile_interval_sec > 0:
                    next_underutilized_reconcile_at = _next_deadline(
                        now,
                        underutilized_reconcile_interval_sec,
                    )
                if changed:
                    continue

            if plan_refresh_interval_sec <= 0 or now >= next_full_reconcile_at:
                recheck_hashes = (
                    set(state.ready_set)
                    | set(state.blocked_hashes)
                    | set(state.failed_hashes)
                    | set(state.in_progress_external_hashes)
                )
                changed = _recheck_hashes(state, recheck_hashes)
                if plan_refresh_interval_sec > 0:
                    next_full_reconcile_at = _next_deadline(
                        now,
                        plan_refresh_interval_sec,
                    )
                if changed:
                    continue

            active_ready = bool(active_scope & state.ready_set)
            active_external = bool(active_scope & state.in_progress_external_hashes)
            active_blocked = active_scope & state.blocked_hashes
            active_failed = active_scope & state.failed_hashes

            if not state.inflight and not active_ready:
                stale_reconciled = _reconcile_external_stale(
                    state,
                    external_hashes=active_scope & state.in_progress_external_hashes,
                    stale_timeout_sec=FURU_CONFIG.stale_timeout,
                )
                if stale_reconciled and _recheck_hashes(
                    state,
                    active_scope & state.in_progress_external_hashes,
                ):
                    continue

                forced_hashes = active_scope & (
                    state.in_progress_external_hashes
                    | state.blocked_hashes
                    | state.failed_hashes
                    | state.ready_set
                )
                if forced_hashes and _recheck_hashes(state, forced_hashes):
                    continue

                active_ready = bool(active_scope & state.ready_set)
                active_external = bool(active_scope & state.in_progress_external_hashes)
                active_blocked = active_scope & state.blocked_hashes
                active_failed = active_scope & state.failed_hashes

                if (
                    not active_ready
                    and not active_external
                    and (active_blocked or active_failed)
                ):
                    sample = _sample_nodes(state, active_blocked | active_failed)
                    raise RuntimeError(
                        f"run_local stalled with no progress; remaining nodes: {sample}"
                    )

            now = time.time()
            wait_timeout = poll_interval_sec
            if state.in_progress_external_hashes and external_poll_interval_sec > 0:
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, next_external_poll_at - now),
                )
            if len(state.inflight) < max_workers and (
                underutilized_reconcile_interval_sec > 0
            ):
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, next_underutilized_reconcile_at - now),
                )
            if plan_refresh_interval_sec > 0:
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, next_full_reconcile_at - now),
                )

            if state.inflight:
                wait(
                    state.inflight.values(),
                    timeout=wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                continue

            if wait_timeout > 0:
                time.sleep(wait_timeout)
