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
from typing import assert_never

from websockets.sync.client import ClientConnection

from furu.config import _Config, get_config
from furu.core import Spec
from furu.dag import Dag, DagNode
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

_RUNNING_ELSEWHERE_POLL_INTERVAL_S = 5.0


@dataclass(frozen=True, slots=True)
class FailedJob:
    failed_attempts: int
    node: DagNode
    error: str


@dataclass(slots=True, kw_only=True)
class Run(Dag):
    """One plan of a worker run, driven over one connection to the manager."""

    backends: dict[str, WorkerBackend]
    provenance: SubmitProvenance
    code: CodeLocation
    config: _Config
    execution_log: Path
    max_retries_per_object: int = field(
        default_factory=lambda: get_config().worker.max_retries_per_object
    )
    placement: dict[str, dict[str, Batch | None]] = field(default_factory=dict)
    running: dict[str, tuple[DagNode, float]] = field(default_factory=dict)
    # Locked by another run; polled until the lock goes, then ready again.
    running_elsewhere: dict[str, DagNode] = field(default_factory=dict)
    failed: dict[str, FailedJob] = field(default_factory=dict)
    batch_keys: list[object] = field(default_factory=list)

    def _check_new(self, nodes: Sequence[DagNode]) -> None:
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
        waiting = 0
        while self.ready:  # completing one can make its dependents ready
            object_id = next(iter(self.ready))
            node = self.ready.pop(object_id)
            match node.obj.status:
                case "done":
                    self.complete(node)
                    logger.info("found %s done", node.obj._log_label)
                case "running":
                    self.running_elsewhere[object_id] = node
                    waiting += 1
                case "missing" | "failed" | "stale":
                    self.running[object_id] = (node, time.monotonic())
                    submissions.append(self._submission(node))
                case unreachable:
                    assert_never(unreachable)
        if waiting:
            logger.info(
                "waiting on %d spec%s running elsewhere",
                waiting,
                "" if waiting == 1 else "s",
            )
        return submissions

    def _submission(self, node: DagNode) -> Submission:
        obj = node.obj
        previous = self.failed.get(obj.object_id)
        artifact = ArtifactSpec.from_furu(obj)
        return Submission(
            job=Job(
                artifacts=[artifact],
                run_logs=[run_log_path_in(obj._base_dir)],
                attempt=(previous.failed_attempts if previous else 0) + 1,
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

    def poll_running_elsewhere(self) -> None:
        for object_id, node in list(self.running_elsewhere.items()):
            if node.obj.status != "running":
                self.ready[object_id] = self.running_elsewhere.pop(object_id)

    def drive(self, connection: ClientConnection) -> None:
        """Submit ready work and apply results until the run ends."""
        pools = {key: backend.pool() for key, backend in self.backends.items()}
        # The first message always goes, so the manager knows who we are.
        jobs = self.take_submissions()
        connection.send(SubmitMessage(pools=pools, jobs=jobs).model_dump_json())
        next_poll = time.monotonic() + _RUNNING_ELSEWHERE_POLL_INTERVAL_S
        while self.ready or self.running or self.running_elsewhere:
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
                self.poll_running_elsewhere()
                next_poll = time.monotonic() + _RUNNING_ELSEWHERE_POLL_INTERVAL_S
            if jobs := self.take_submissions():
                connection.send(SubmitMessage(pools=pools, jobs=jobs).model_dump_json())
        self.finish()

    def job_result(self, object_id: str, result: JobResult) -> None:
        running = self.running.pop(object_id, None)
        if running is None:
            logger.info("ignoring result for %s: not running", object_id)
            return
        node, started_at = running
        match result:
            case JobCompletedResult():
                self.failed.pop(object_id, None)
                self.complete(node)
                logger.info(
                    "completed %s ok · %s",
                    node.obj._log_label,
                    format_duration(time.monotonic() - started_at),
                )
            case JobFailedResult(error=error):
                previous = self.failed.get(object_id)
                failed_attempts = (previous.failed_attempts if previous else 0) + 1
                self.failed[object_id] = FailedJob(
                    failed_attempts=failed_attempts, node=node, error=error
                )
                will_retry = failed_attempts <= self.max_retries_per_object
                if will_retry:
                    self.ready[object_id] = node
                # The traceback lives in run.log; repeats say so instead.
                summary = error_summary(error)
                if previous and error_summary(previous.error) == summary:
                    summary = "same error"
                logger.log(
                    logging.WARNING if will_retry else logging.ERROR,
                    "failed %s · attempt %d/%d · %s",
                    node.obj._log_label,
                    failed_attempts,
                    self.max_retries_per_object,
                    summary,
                    extra={"path": run_log_path_in(node.obj._base_dir)},
                )
            case JobBlockedResult(dependencies=dependencies):
                try:
                    self.block(node, [Spec.from_artifact(dep) for dep in dependencies])
                except Exception as exc:  # noqa: BLE001 -- runs user runs_on, accepts and batch fns
                    self.fail(f"{type(exc).__name__}: {exc}")
                logger.info(
                    "blocked %s · %s",
                    node.obj._log_label,
                    missing_dependencies(len(dependencies)),
                )
            case _:
                assert_never(result)
        retrying = sum(
            record.failed_attempts <= self.max_retries_per_object
            for record in self.failed.values()
        )
        parts = [f"{len(self.running)} running"]
        if self.ready:
            parts.append(f"{len(self.ready)} ready")
        if self.blocked:
            parts.append(f"{len(self.blocked)} blocked")
        if failed := len(self.failed) - retrying:
            parts.append(f"{failed} failed")
        if retrying:
            parts.append(f"{retrying} retrying")
        logger.info(
            "progress %d/%d · %s",
            len(self.completed),
            len(self.nodes_by_id),
            " · ".join(parts),
        )

    def fail(self, message: str) -> None:
        logger.error("run failed · %s", message, extra={"path": self.execution_log})
        raise RuntimeError(message)

    def finish(self) -> None:
        """Nothing is ready or running: succeed, or say what failed and why."""
        terminal = [
            record
            for record in self.failed.values()
            if record.failed_attempts > self.max_retries_per_object
        ]
        if not terminal and not self.blocked:
            logger.info("run finished ok")
            return

        def plural(count: int) -> str:
            return f"{count} spec{'' if count == 1 else 's'}"

        lines = [f"{plural(len(terminal))} failed, {len(self.blocked)} blocked"]
        for record in terminal:
            obj = record.node.obj
            lines.append(
                f"  {obj._log_label} · {record.failed_attempts} attempt"
                f"{'' if record.failed_attempts == 1 else 's'} · "
                f"{error_summary(record.error)} → "
                f"{_display_path(run_log_path_in(obj._base_dir))}"
            )
        lines += [
            f"  {node.obj._log_label} · blocked" for node in self.blocked.values()
        ]

        headline = ["run failed"]
        if terminal:
            headline.append(f"{plural(len(terminal))} failed")
        if self.blocked:
            headline.append(f"{plural(len(self.blocked))} blocked")
        if len(terminal) == 1:
            (record,) = terminal
            headline += [record.node.obj._log_label, error_summary(record.error)]
        logger.error(" · ".join(headline), extra={"path": self.execution_log})
        raise RuntimeError("\n".join(lines))
