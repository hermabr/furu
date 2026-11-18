"""The script's half of a worker run: the graph, retries and lock polling.

Everything that needs your code (``runs_on``, ``accepts``, batch functions)
runs here; the manager gets plain data.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, NoReturn, assert_never

from websockets.sync.client import ClientConnection

from furu import plan
from furu.config import _Config, get_config
from furu.core import Spec, working_folder
from furu.dag import Dag, DagNode
from furu.dependencies import missing_dependencies
from furu.locking import is_active_lock
from furu.logging import _display_path, get_logger
from furu.provenance import SubmitProvenance
from furu.snapshot import CodeLocation
from furu.storage._layout import lock_path_in, log_path_in
from furu.storage.stored import Stored
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


def _running_elsewhere(obj: Spec) -> bool:
    """Whether a live worker holds the working folder of obj running alone.

    A batch's lock is named by all its specs, so a spec that another run
    computes in a batch is not seen here and may be computed twice.
    """
    return is_active_lock(lock_path_in(obj._class_dir, obj._artifact_hash))


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
    # Polled until no other run holds their lock, then ready again: specs
    # locked elsewhere, and ones that found missing only specs this run sees
    # built (their worker's filesystem may not show those yet).
    waiting: dict[str, DagNode] = field(default_factory=dict)
    # Gave up after max_retries_per_object retries.
    failed: dict[str, DagNode] = field(default_factory=dict)
    # Each failed attempt's error, per spec not yet completed, and the log of
    # the last one: its batch's.
    failures: dict[str, list[str]] = field(default_factory=dict)
    logs: dict[str, Path] = field(default_factory=dict)
    batch_ids: dict[object, str] = field(default_factory=dict)
    # The stored names of the classes this launch asked about.
    asked: frozenset[str] = frozenset()

    # A worker's filesystem can take about a minute to show a version another
    # node just published (NFS caches directory listings).
    unseen_grace_s: ClassVar[float] = 120.0

    def _check_new(self, nodes: Sequence[DagNode]) -> None:
        unsatisfiable = []
        for node in nodes:
            pools: dict[str, Batch | None] = {}
            for key, backend in self.backends.items():
                if not can_run(backend, node.obj):
                    continue
                group = self.batch_group(node, backend.worker)
                pools[key] = (
                    None
                    if group is None
                    else Batch(
                        # A batch holds specs from one plan, which its run names.
                        batch=self.batch_ids.setdefault(
                            (group[0], node.plan), f"b{len(self.batch_ids)}"
                        ),
                        cap=group[1],
                    )
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

    def take_submissions(self) -> list[Submission]:
        """Ready specs to send; ones done or locked elsewhere don't need us."""
        submissions: list[Submission] = []
        locked = 0
        look = Stored()
        while self.ready:  # completing one can make its dependents ready
            object_id = next(iter(self.ready))
            node = self.ready.pop(object_id)
            if look.row(node.obj) is not None:
                self.complete(node)
                logger.info("found %s done", node.obj._log_label)
            elif _running_elsewhere(node.obj):
                self.waiting[object_id] = node
                locked += 1
            else:
                self.running[object_id] = (node, time.monotonic())
                submissions.append(self._submission(node))
        if locked:
            logger.info(
                "waiting on %d spec%s running elsewhere",
                locked,
                "" if locked == 1 else "s",
            )
        return submissions

    def _submission(self, node: DagNode) -> Submission:
        obj = node.obj
        return Submission(
            job=Job(
                specs=[obj.object_id],
                lines=list(map(plan.dump, plan.lines([obj]))),
                plan=node.plan,
                class_dir=obj._class_dir,
                execution_log=self.execution_log,
                provenance=self.provenance,
                process=ProcessSettings.from_metadata(obj._metadata),
                config=self.config,
                code=self.code,
                asked=self.asked,
            ),
            pools=self.placement[obj.object_id],
            throttle=None
            if (throttle := obj.throttle) is None
            else JobThrottle(key=obj._stored_name, max_running=throttle.max_running),
        )

    def poll_waiting(self) -> None:
        for object_id, node in list(self.waiting.items()):
            if not _running_elsewhere(node.obj):
                self.ready[object_id] = self.waiting.pop(object_id)

    def drive(self, connection: ClientConnection) -> None:
        """Submit ready work and apply results until the run ends."""
        pools = {key: backend.pool() for key, backend in self.backends.items()}
        # The first message always goes, so the manager knows who we are.
        jobs = self.take_submissions()
        connection.send(SubmitMessage(pools=pools, jobs=jobs).model_dump_json())
        next_poll = time.monotonic() + _POLL_INTERVAL_S
        while self.ready or self.running or self.waiting:
            try:
                raw = connection.recv(timeout=max(0.0, next_poll - time.monotonic()))
            except TimeoutError:
                raw = None
            if raw is not None:
                match manager_message_adapter.validate_json(raw):
                    case ResultMessage(object_ids=object_ids, result=result):
                        self.batch_result(object_ids, result)
                    case WarningMessage(reason=reason):
                        logger.warning("%s", reason)
                    case ErrorMessage(reason=reason):
                        self.fail(reason)
                    case _ as unreachable:
                        assert_never(unreachable)
            if time.monotonic() >= next_poll:
                self.poll_waiting()
                next_poll = time.monotonic() + _POLL_INTERVAL_S
            if jobs := self.take_submissions():
                connection.send(SubmitMessage(pools=pools, jobs=jobs).model_dump_json())
        self.finish()

    def batch_result(self, object_ids: Sequence[str], result: JobResult) -> None:
        """Apply the result of one create() call to each spec it ran."""
        ran = [self.running[i][0].obj for i in object_ids if i in self.running]
        log = log_path_in(working_folder(ran)) if ran else None
        for object_id in object_ids:
            self.job_result(object_id, result, log=log)

    def job_result(
        self, object_id: str, result: JobResult, *, log: Path | None = None
    ) -> None:
        """Apply a create() call's result to one spec it ran.

        ``log`` is that call's log; a spec that ran alone has its own.
        """
        running = self.running.pop(object_id, None)
        if running is None:
            logger.info("ignoring result for %s: not running", object_id)
            return
        node, started_at = running
        log = self.logs[object_id] = log or log_path_in(node.obj._attempt_dir)
        match result:
            case JobCompletedResult():
                self.complete(node)
                logger.info(
                    "completed %s ok · %s",
                    node.obj._log_label,
                    format_duration(time.monotonic() - started_at),
                )
            case JobFailedResult(error=error):
                node.unseen_since.clear()  # its earlier misses are behind it
                errors = self.failures.setdefault(object_id, [])
                errors.append(error)
                will_retry = len(errors) <= self.max_retries_per_object
                (self.ready if will_retry else self.failed)[object_id] = node
                # The traceback lives in the run's log; repeats say so instead.
                summary = error_summary(error)
                if len(errors) > 1 and error_summary(errors[-2]) == summary:
                    summary = "same error"
                logger.log(
                    logging.WARNING if will_retry else logging.ERROR,
                    "failed %s · attempt %d/%d · %s",
                    node.obj._log_label,
                    len(errors),
                    self.max_retries_per_object,
                    summary,
                    extra={"path": log},
                )
            case JobBlockedResult(dependencies=dependencies):
                try:
                    found = plan.Specs(map(json.loads, result.lines))
                    why = self.block(
                        node, [found[dep] for dep in dependencies], result.plan
                    )
                except Exception as exc:  # noqa: BLE001 -- runs user runs_on, accepts and batch fns
                    self.fail(f"{type(exc).__name__}: {exc}")
                logger.info(
                    "blocked %s · %s",
                    node.obj._log_label,
                    missing_dependencies(len(dependencies)),
                )
                if why is not None:
                    if dependencies:
                        why += (
                            "; its worker may not see the storage it was written "
                            "to, or may be running different code than this run"
                        )
                    self.failures.setdefault(object_id, []).append(why)
                    self.failed[object_id] = node
                    logger.error(
                        "failed %s · %s",
                        node.obj._log_label,
                        why,
                        extra={"path": log},
                    )
                elif object_id in self.ready:
                    # Rerunning at once would likely miss the same specs again.
                    self.waiting[object_id] = self.ready.pop(object_id)
            case _:
                assert_never(result)
        parts = [f"{len(self.running)} running"]
        if self.ready:
            parts.append(f"{len(self.ready)} ready")
        if self.blocked:
            parts.append(f"{len(self.blocked)} blocked")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if retrying := len(self.failures) - len(self.failed):
            parts.append(f"{retrying} retrying")
        logger.info(
            "progress %d/%d · %s",
            len(self.completed),
            len(self.nodes_by_id),
            " · ".join(parts),
        )

    def complete(self, node: DagNode) -> None:
        # Here or found done elsewhere, it is no longer retrying.
        self.failures.pop(node.obj.object_id, None)
        Dag.complete(self, node)  # slots dataclasses break zero-argument super()

    def fail(self, message: str) -> NoReturn:
        logger.error("run failed · %s", message, extra={"path": self.execution_log})
        raise RuntimeError(message)

    def finish(self) -> None:
        """Nothing is ready or running: succeed, or say what failed and why."""
        if not self.failed and not self.blocked:
            logger.info("run finished ok")
            return

        def plural(count: int) -> str:
            return f"{count} spec{'' if count == 1 else 's'}"

        # In order of first failure, like the run's log.
        failed = [
            (self.failed[object_id].obj, errors)
            for object_id, errors in self.failures.items()
            if object_id in self.failed
        ]
        lines = [f"{plural(len(failed))} failed, {len(self.blocked)} blocked"]
        for obj, errors in failed:
            lines.append(
                f"  {obj._log_label} · {len(errors)} attempt"
                f"{'' if len(errors) == 1 else 's'} · "
                f"{error_summary(errors[-1])} → "
                f"{_display_path(self.logs[obj.object_id])}"
            )
        lines += [
            f"  {node.obj._log_label} · blocked" for node in self.blocked.values()
        ]

        headline = ["run failed"]
        if failed:
            headline.append(f"{plural(len(failed))} failed")
        if self.blocked:
            headline.append(f"{plural(len(self.blocked))} blocked")
        if len(failed) == 1:
            ((obj, errors),) = failed
            headline += [obj._log_label, error_summary(errors[-1])]
        logger.error(" · ".join(headline), extra={"path": self.execution_log})
        raise RuntimeError("\n".join(lines))
