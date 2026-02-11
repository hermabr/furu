from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

from ..config import FURU_CONFIG
from ..core import Furu
from ..errors import FuruComputeError, FuruError
from ..runtime.logging import enter_holder
from ..storage.state import StateManager
from .context import EXEC_CONTEXT, ExecContext
from .plan import DependencyPlan, PlanNode, build_plan, ready_todo
from .plan_utils import reconcile_or_timeout_in_progress


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


def _run_node(node: PlanNode) -> None:
    token = EXEC_CONTEXT.set(
        ExecContext(
            mode="executor",
            spec_key=node.executor_key,
            backend="local",
            current_node_hash=node.obj.furu_hash,
        )
    )
    try:
        with enter_holder(node.obj):
            node.obj.get(force=True)
    finally:
        EXEC_CONTEXT.reset(token)


@dataclass(frozen=True)
class _InProgressOwnership:
    self_hashes: set[str]
    external_hashes: set[str]


def _in_progress_ownership(
    plan: "DependencyPlan",
    *,
    inflight_hashes: set[str],
) -> _InProgressOwnership:
    self_hashes = {digest for digest in inflight_hashes if digest in plan.nodes}
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


def run_local(
    roots: list[Furu],
    *,
    max_workers: int = 8,
    window_size: str | int = "bfs",
    poll_interval_sec: float = 0.25,
    plan_refresh_interval_sec: float = 60.0,
) -> None:
    if not roots:
        return
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if plan_refresh_interval_sec < 0:
        raise ValueError("plan_refresh_interval_sec must be >= 0")

    window = _normalize_window_size(window_size, len(roots))
    active_indices = list(range(min(window, len(roots))))
    next_index = len(active_indices)
    inflight: dict[str, Future[None]] = {}
    completed_hashes: set[str] = set()
    retry_attempts: dict[str, int] = {}
    plan: DependencyPlan | None = None
    plan_signature: tuple[int, ...] = tuple()
    next_plan_refresh_at = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            active_roots = [roots[index] for index in active_indices]
            active_signature = tuple(active_indices)
            now = time.time()
            needs_plan_refresh = (
                plan is None
                or plan_signature != active_signature
                or now >= next_plan_refresh_at
            )
            if needs_plan_refresh:
                plan = build_plan(active_roots, completed_hashes=completed_hashes)
                plan_signature = active_signature
                completed_hashes.update(
                    digest
                    for digest, node in plan.nodes.items()
                    if node.status == "DONE"
                )
                next_plan_refresh_at = _next_refresh_deadline(
                    now,
                    plan_refresh_interval_sec,
                )
            if plan is None:
                raise RuntimeError(
                    "internal error: local executor plan not initialized"
                )

            ready = [digest for digest in ready_todo(plan) if digest not in inflight]
            available = max_workers - len(inflight)
            for digest in ready[:available]:
                node = plan.nodes[digest]
                inflight[digest] = executor.submit(_run_node, node)

            completed = [digest for digest, future in inflight.items() if future.done()]
            for digest in completed:
                future = inflight.pop(digest)
                try:
                    future.result()
                except Exception as exc:
                    if isinstance(exc, FuruComputeError):
                        compute_error = exc
                        wrapped_exc: Exception | None = None
                    elif isinstance(exc, FuruError):
                        raise
                    else:
                        node = plan.nodes.get(digest)
                        if node is None:
                            raise
                        state_path = StateManager.get_state_path(
                            node.obj._base_furu_dir()
                        )
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
                        continue
                    if wrapped_exc is not None:
                        raise compute_error from wrapped_exc
                    raise compute_error
                completed_hashes.add(digest)
                node = plan.nodes.get(digest)
                if node is not None:
                    node.status = "DONE"
                retry_attempts.pop(digest, None)

            if not FURU_CONFIG.retry_failed:
                failed = [
                    node
                    for digest, node in plan.nodes.items()
                    if node.status == "FAILED" and digest not in inflight
                ]
                if failed:
                    names = ", ".join(
                        f"{node.obj.__class__.__name__}({node.obj.furu_hash})"
                        for node in failed
                    )
                    raise RuntimeError(
                        f"Cannot run local executor with failed dependencies: {names}"
                    )

            if completed:
                continue

            # Avoid a busy-spin loop while waiting for long-running tasks.
            if inflight and not completed:
                wait(
                    inflight.values(),
                    timeout=poll_interval_sec,
                    return_when=FIRST_COMPLETED,
                )
                continue

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

            if not active_indices and not inflight and next_index >= len(roots):
                return

            if not inflight and not ready:
                ownership = _in_progress_ownership(
                    plan,
                    inflight_hashes=set(inflight),
                )
                if ownership.self_hashes or ownership.external_hashes:
                    stale_detected = reconcile_or_timeout_in_progress(
                        plan,
                        stale_timeout_sec=FURU_CONFIG.stale_timeout,
                    )
                    if stale_detected:
                        next_plan_refresh_at = 0.0
                        continue
                    time.sleep(poll_interval_sec)
                    continue
                todo_nodes = [
                    node for node in plan.nodes.values() if node.status == "TODO"
                ]
                if todo_nodes:
                    sample = ", ".join(
                        f"{node.obj.__class__.__name__}({node.obj.furu_hash})"
                        for node in todo_nodes[:3]
                    )
                    raise RuntimeError(
                        "run_local stalled with no progress; "
                        f"remaining TODO nodes: {sample}"
                    )
                time.sleep(poll_interval_sec)
