from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..adapters import SubmititAdapter
from ..adapters.submitit import SubmititJob
from ..config import FURU_CONFIG
from ..core import Furu
from ..errors import FuruExecutionError
from ..storage.state import StateManager, _FuruState, _StateResultFailed
from .plan import DependencyPlan, build_plan, topo_order_todo
from .slurm_spec import SlurmSpecExtraValue
from .submitit_factory import make_executor_for_spec


@dataclass
class SlurmDagSubmission:
    plan: DependencyPlan
    job_id_by_hash: dict[str, str]
    root_job_ids: dict[str, str]
    run_id: str


def _job_id_from_state(obj: Furu, directory: Path | None = None) -> str | None:
    state = obj.get_state(directory)
    attempt = state.attempt
    if attempt is None:
        return None
    job_id = attempt.scheduler.get("job_id")
    if job_id is None:
        return None
    return str(job_id)


def _attempt_is_terminal(obj: Furu, directory: Path | None = None) -> bool:
    state = obj.get_state(directory)
    attempt = state.attempt
    if attempt is None:
        return False
    return attempt.status in StateManager.TERMINAL_STATUSES


def _set_submitit_job_id(directory: Path, job_id: str) -> bool:
    updated = False

    def mutate(state: _FuruState) -> bool:
        nonlocal updated
        attempt = state.attempt
        if attempt is None:
            return False
        if attempt.backend != "submitit":
            return False
        if (
            attempt.status not in {"queued", "running"}
            and attempt.status not in StateManager.TERMINAL_STATUSES
        ):
            return False
        existing = attempt.scheduler.get("job_id")
        if existing == job_id:
            return False
        attempt.scheduler["job_id"] = job_id
        updated = True
        return True

    StateManager.update_state(directory, mutate)
    return updated


def _wait_for_job_id(
    obj: Furu,
    adapter: SubmititAdapter,
    job: SubmititJob | None,
    *,
    timeout_sec: float = 15.0,
    poll_interval_sec: float = 0.25,
) -> str:
    deadline = time.time() + timeout_sec
    directory = obj._base_furu_dir()
    last_job_id: str | None = None

    while True:
        job_id = _job_id_from_state(obj, directory)
        if job_id:
            if job is None:
                return job_id
            adapter_job_id = adapter.get_job_id(job)
            if adapter_job_id is None or str(adapter_job_id) == job_id:
                return job_id
            last_job_id = str(adapter_job_id)
            _set_submitit_job_id(directory, last_job_id)
            state_job_id = _job_id_from_state(obj, directory)
            if state_job_id is not None and state_job_id == last_job_id:
                return state_job_id
            if _attempt_is_terminal(obj, directory):
                return last_job_id

        if job is None:
            job = adapter.load_job(directory)

        if job is not None:
            job_id = adapter.get_job_id(job)
            if job_id:
                last_job_id = job_id
                _set_submitit_job_id(directory, job_id)
                state_job_id = _job_id_from_state(obj, directory)
                if state_job_id is not None and state_job_id == job_id:
                    return state_job_id
                if _attempt_is_terminal(obj, directory):
                    return str(job_id)

        if time.time() >= deadline:
            suffix = f" Last seen job_id={last_job_id}." if last_job_id else ""
            raise TimeoutError(
                "Timed out waiting for submitit job_id for "
                f"{obj.__class__.__name__} ({obj.furu_hash}).{suffix}"
            )

        time.sleep(poll_interval_sec)


def _job_id_for_in_progress(obj: Furu) -> str:
    state = obj.get_state()
    attempt = state.attempt
    if attempt is None:
        raise RuntimeError(
            "Cannot wire Slurm DAG dependency for IN_PROGRESS "
            f"{obj.__class__.__name__} ({obj.furu_hash}) without an attempt."
        )
    if attempt.backend != "submitit":
        raise FuruExecutionError(
            "Cannot wire afterok dependencies to non-submitit in-progress nodes. "
            "Use pool mode or wait until completed."
        )

    # If the dependency has already become terminal and failed (or otherwise did not
    # succeed), wiring `afterok` would permanently block dependents.
    if isinstance(state.result, _StateResultFailed) or (
        attempt.status in StateManager.TERMINAL_STATUSES and attempt.status != "success"
    ):
        raise FuruExecutionError(
            "Cannot wire afterok dependency to a terminal non-success dependency. "
            f"Dependency {obj.__class__.__name__} ({obj.furu_hash}) status={attempt.status}."
        )

    job_id = attempt.scheduler.get("job_id")
    if job_id:
        resolved = str(job_id)
    else:
        adapter = SubmititAdapter(executor=None)
        resolved = _wait_for_job_id(obj, adapter, None)

    # Re-check after waiting: the attempt could flip to terminal while we're
    # retrieving job_id. If it ended non-success, fail fast instead of wiring
    # dependents to an `afterok` that will never unblock.
    state2 = obj.get_state()
    attempt2 = state2.attempt
    if attempt2 is not None and attempt2.status in StateManager.TERMINAL_STATUSES:
        if attempt2.status != "success" or isinstance(
            state2.result, _StateResultFailed
        ):
            raise FuruExecutionError(
                "Cannot wire afterok dependency: dependency became terminal and did not succeed. "
                f"Dependency {obj.__class__.__name__} ({obj.furu_hash}) status={attempt2.status} "
                f"job_id={resolved}."
            )

    return resolved


def _make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:6]
    return f"{stamp}-{token}"


def submit_slurm_dag(
    roots: list[Furu],
    *,
    submitit_root: Path | None = None,
) -> SlurmDagSubmission:
    run_id = _make_run_id()
    plan = build_plan(roots)
    failed = [node for node in plan.nodes.values() if node.status == "FAILED"]
    if failed:
        names = ", ".join(
            f"{node.obj.__class__.__name__}({node.obj.furu_hash})" for node in failed
        )
        raise RuntimeError(f"Cannot submit slurm DAG with failed dependencies: {names}")

    order = topo_order_todo(plan)
    job_id_by_hash: dict[str, str] = {}
    root_job_ids: dict[str, str] = {}

    root_hashes = {root.furu_hash for root in roots}

    for digest in order:
        node = plan.nodes[digest]
        dep_job_ids: list[str] = []
        for dep_hash in sorted(node.deps_pending):
            dep_node = plan.nodes[dep_hash]
            if dep_node.status == "IN_PROGRESS":
                dep_job_ids.append(_job_id_for_in_progress(dep_node.obj))
            elif dep_node.status == "TODO":
                dep_job_ids.append(job_id_by_hash[dep_hash])

        spec_key = node.executor_key
        spec = node.executor
        executor = make_executor_for_spec(
            spec_key,
            spec,
            kind="nodes",
            submitit_root=submitit_root,
            run_id=run_id,
        )
        if dep_job_ids:
            dependency = "afterok:" + ":".join(dep_job_ids)
            slurm_params: dict[str, SlurmSpecExtraValue] = {"dependency": dependency}
            if spec.extra:
                extra_params = spec.extra.get("slurm_additional_parameters")
                if extra_params is not None:
                    if not isinstance(extra_params, Mapping):
                        raise TypeError(
                            "slurm_additional_parameters must be a mapping when provided."
                        )
                    slurm_params = {
                        **dict(extra_params),
                        "dependency": dependency,
                    }
            executor.update_parameters(slurm_additional_parameters=slurm_params)

        adapter = SubmititAdapter(executor)
        job = node.obj._submit_once(
            adapter,
            directory=node.obj._base_furu_dir(),
            on_job_id=None,
            allow_failed=FURU_CONFIG.retry_failed,
        )
        job_id = _wait_for_job_id(node.obj, adapter, job)
        job_id_by_hash[digest] = job_id
        if digest in root_hashes:
            root_job_ids[digest] = job_id

    for root in roots:
        digest = root.furu_hash
        if digest in root_job_ids:
            continue
        node = plan.nodes.get(digest)
        if node is None:
            continue
        if node.status == "IN_PROGRESS":
            root_job_ids[digest] = _job_id_for_in_progress(node.obj)

    return SlurmDagSubmission(
        plan=plan,
        job_id_by_hash=job_id_by_hash,
        root_job_ids=root_job_ids,
        run_id=run_id,
    )
