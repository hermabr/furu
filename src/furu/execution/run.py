"""The script's half of a worker run: the graph, retries and lock polling.

Everything that needs your code (``runs_on``, ``accepts``, batch functions)
runs here; the manager gets plain data.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, assert_never

from websockets.sync.client import ClientConnection

from furu.config import _Config, get_config
from furu.core import Spec
from furu.dag import Dag, DagNode, State
from furu.dependencies import missing_dependencies
from furu.logging import _display_path, get_logger
from furu.metadata import ArtifactSpec
from furu.provenance import SubmitProvenance
from furu.snapshot import CodeLocation
from furu.storage._layout import run_log_path_in
from furu.utils import error_summary, format_duration
from furu.worker.backends.protocol import WorkerBackend, can_run
from furu.worker.messages import (
    Batch,
    ErrorMessage,
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    JobThrottle,
    ProcessSettings,
    ResultMessage,
    Submission,
    SubmitMessage,
    WarningMessage,
    manager_message_adapter,
)

logger = get_logger("run")

_POLL_INTERVAL_S = 5.0


# A worker's filesystem can take about a minute to show a version another
# node just published (NFS caches directory listings).
_UNSEEN_GRACE_S = 120.0
_UNSEEN_HINT = (
    "; its worker may not see the storage it was written to, "
    "or may be running different code than this run"
)


@dataclass(slots=True, kw_only=True)
class Run:
    """One plan of a worker run, driven over one connection to the manager."""

    backends: dict[str, WorkerBackend]
    provenance: SubmitProvenance
    code: CodeLocation
    config: _Config
    execution_log: Path
    max_retries_per_object: int = field(
        default_factory=lambda: get_config().worker.max_retries_per_object
    )
    unseen_grace_s: float = _UNSEEN_GRACE_S
    dag: Dag = field(init=False)
    placement: dict[str, dict[str, Batch | None]] = field(default_factory=dict)
    batch_keys: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dag = Dag(check_new=self._place)

    def _place(self, nodes: Sequence[DagNode]) -> None:
        """Note the pools each new node can run on; fail if one has none."""
        unsatisfiable = []
        for node in nodes:
            pools: dict[str, Batch | None] = {}
            for key, backend in self.backends.items():
                if not can_run(backend, node.obj):
                    continue
                group = node.batch_group(backend.worker)
                pools[key] = (
                    None
                    if group is None
                    else Batch(batch=self._batch_id(group[0]), cap=group[1])
                )
            if not pools:
                unsatisfiable.append(node.obj._log_label)
            self.placement[node.obj.object_id] = pools
        if unsatisfiable:
            workers = ", ".join(
                str(backend.worker) for backend in self.backends.values()
            )
            raise RuntimeError(
                "no worker pool can run: "
                + ", ".join(sorted(unsatisfiable))
                + f"; workers: {workers or 'none'}"
            )

    def _batch_id(self, key: object) -> str:
        # Batch keys hold metadata dicts, so they compare but don't hash.
        if key not in self.batch_keys:
            self.batch_keys.append(key)
        return f"b{self.batch_keys.index(key)}"

    def take_submissions(self) -> list[Submission]:
        """Ready specs to send; ones done or locked elsewhere don't need us."""
        submissions: list[Submission] = []
        locked = 0
        ready = self.dag.nodes(State.READY)
        while ready:  # completing one can make its dependents ready
            node = next(iter(ready.values()))
            match self.dag.start(node):
                case State.RUNNING:
                    submissions.append(self._submission(node))
                case State.LOCKED:
                    locked += 1
                case State.DONE:
                    pass
                case other:
                    raise AssertionError(f"started {node.obj._log_label}: {other}")
        if locked:
            logger.info(
                "waiting on %d spec%s running elsewhere",
                locked,
                "" if locked == 1 else "s",
            )
        return submissions

    def _submission(self, node: DagNode) -> Submission:
        obj = node.obj
        artifact = ArtifactSpec.from_furu(obj)
        return Submission(
            job=Job(
                artifacts=[artifact],
                run_logs=[run_log_path_in(obj._base_dir)],
                attempt=node.attempts + 1,
                execution_log=self.execution_log,
                provenance=self.provenance,
                process=ProcessSettings.from_metadata(obj._metadata),
                config=self.config,
                code=self.code,
            ),
            pools=self.placement[obj.object_id],
            throttle=None
            if (throttle := obj.throttle) is None
            else JobThrottle(
                key=artifact.fully_qualified_name, max_running=throttle.max_running
            ),
        )

    def drive(self, connection: ClientConnection) -> None:
        """Submit ready work and apply results until the run ends."""
        pools = {key: backend.pool() for key, backend in self.backends.items()}
        # The first message always goes, so the manager knows who we are.
        jobs = self.take_submissions()
        connection.send(SubmitMessage(pools=pools, jobs=jobs).model_dump_json())
        next_poll = time.monotonic() + _POLL_INTERVAL_S
        while self.dag.active():
            try:
                raw = connection.recv(timeout=max(0.0, next_poll - time.monotonic()))
            except TimeoutError:
                raw = None
            if raw is not None:
                match manager_message_adapter.validate_json(raw):
                    case ResultMessage(object_ids=object_ids, result=result):
                        for object_id in object_ids:
                            self.job_result(object_id, result)
                    case WarningMessage(reason=reason):
                        logger.warning("%s", reason)
                    case ErrorMessage(reason=reason):
                        self.fail(reason)
                    case _ as unreachable:
                        assert_never(unreachable)
            if time.monotonic() >= next_poll:
                self.dag.poll()
                next_poll = time.monotonic() + _POLL_INTERVAL_S
            if jobs := self.take_submissions():
                connection.send(SubmitMessage(pools=pools, jobs=jobs).model_dump_json())
        self.finish()

    def job_result(self, object_id: str, result: JobResult) -> None:
        node = self.dag.nodes(State.RUNNING).get(object_id)
        if node is None:
            logger.info("ignoring result for %s: not running", object_id)
            return
        match result:
            case JobCompletedResult():
                self.dag.complete(node)
                logger.info(
                    "completed %s ok · %s",
                    node.obj._log_label,
                    format_duration(time.monotonic() - node.started_at),
                )
            case JobFailedResult(error=error):
                previous = node.error
                will_retry = self.dag.attempt_failed(
                    node, error, max_retries=self.max_retries_per_object
                )
                # The traceback lives in run.log; repeats say so instead.
                summary = error_summary(error)
                if previous is not None and error_summary(previous) == summary:
                    summary = "same error"
                logger.log(
                    logging.WARNING if will_retry else logging.ERROR,
                    "failed %s · attempt %d/%d · %s",
                    node.obj._log_label,
                    node.attempts,
                    self.max_retries_per_object,
                    summary,
                    extra={"path": run_log_path_in(node.obj._base_dir)},
                )
            case JobBlockedResult(dependencies=dependencies):
                try:
                    why = self.dag.block(
                        node,
                        [Spec.from_artifact(dep) for dep in dependencies],
                        unseen_grace_s=self.unseen_grace_s,
                        unseen_hint=_UNSEEN_HINT,
                    )
                except Exception as exc:  # noqa: BLE001 -- runs user runs_on, accepts and batch fns
                    self.fail(f"{type(exc).__name__}: {exc}")
                logger.info(
                    "blocked %s · %s",
                    node.obj._log_label,
                    missing_dependencies(len(dependencies)),
                )
                if why is not None:
                    logger.error(
                        "failed %s · %s",
                        node.obj._log_label,
                        why,
                        extra={"path": run_log_path_in(node.obj._base_dir)},
                    )
            case _:
                assert_never(result)
        dag = self.dag
        parts = [f"{len(dag.nodes(State.RUNNING))} running"]
        for state in (State.READY, State.BLOCKED, State.FAILED):
            if count := len(dag.nodes(state)):
                parts.append(f"{count} {state}")
        retrying = sum(
            1
            for node in dag.nodes_by_id.values()
            if node.attempts and node.state not in (State.DONE, State.FAILED)
        )
        if retrying:
            parts.append(f"{retrying} retrying")
        logger.info(
            "progress %d/%d · %s",
            len(dag.nodes(State.DONE)),
            len(dag.nodes_by_id),
            " · ".join(parts),
        )

    def fail(self, message: str) -> NoReturn:
        logger.error("run failed · %s", message, extra={"path": self.execution_log})
        raise RuntimeError(message)

    def finish(self) -> None:
        """Nothing can move: succeed, or say what failed and why."""
        blocked = self.dag.nodes(State.BLOCKED)
        failed = sorted(
            self.dag.nodes(State.FAILED).values(),
            key=lambda node: node.first_failure or 0,
        )
        if not failed and not blocked:
            logger.info("run finished ok")
            return

        def plural(count: int) -> str:
            return f"{count} spec{'' if count == 1 else 's'}"

        def why(node: DagNode) -> str:
            assert node.error is not None
            return error_summary(node.error)

        lines = [f"{plural(len(failed))} failed, {len(blocked)} blocked"]
        for node in failed:
            attempts = (
                f"{node.attempts} attempt{'' if node.attempts == 1 else 's'} · "
                if node.attempts
                else ""
            )
            lines.append(
                f"  {node.obj._log_label} · {attempts}{why(node)} → "
                f"{_display_path(run_log_path_in(node.obj._base_dir))}"
            )
        lines += [f"  {node.obj._log_label} · blocked" for node in blocked.values()]

        headline = ["run failed"]
        if failed:
            headline.append(f"{plural(len(failed))} failed")
        if blocked:
            headline.append(f"{plural(len(blocked))} blocked")
        if len(failed) == 1:
            headline += [failed[0].obj._log_label, why(failed[0])]
        logger.error(" · ".join(headline), extra={"path": self.execution_log})
        raise RuntimeError("\n".join(lines))
