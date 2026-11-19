"""The script's half of a launch on workers: placement, retries and found specs.

Everything that needs your code (``runs_on``, ``accepts``, batch functions)
runs here. The manager gets plain data: the plan's lines, and for the specs
that run, where and in which batch. It holds the graph and forms the batches;
this half says what happens to an attempt that did not complete.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, NoReturn, assert_never

from websockets.sync.client import ClientConnection

from furu import plan
from furu._batched import batch_of
from furu.config import get_config
from furu.core import Spec, working_folder
from furu.dag import why_stuck
from furu.dependencies import missing_dependencies
from furu.execution.launch import missing
from furu.logging import _display_path, get_logger
from furu.resources import Worker
from furu.storage._layout import log_path_in
from furu.utils import error_summary, format_duration, parse_json_object
from furu.worker.backends.protocol import WorkerBackend, can_run
from furu.worker.messages import (
    Batch,
    ErrorMessage,
    GaveUpMessage,
    Group,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobThrottle,
    Launch,
    ProcessSettings,
    RerunMessage,
    ResultMessage,
    SubmitMessage,
    WarningMessage,
    manager_message_adapter,
)

logger = get_logger("run")

_POLL_INTERVAL_S = 5.0

type _Reply = SubmitMessage | RerunMessage | GaveUpMessage


@dataclass(slots=True, kw_only=True)
class Run:
    """One launch on workers, driven over one connection to the manager."""

    backends: dict[str, WorkerBackend]
    launch: Launch
    max_retries_per_object: int = field(
        default_factory=lambda: get_config().worker.max_retries_per_object
    )
    # Every spec this run asked the manager for, and the ones still to end.
    specs: dict[str, Spec] = field(default_factory=dict)
    left: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)
    # Gave up on, and the ones that wait for those.
    failed: set[str] = field(default_factory=set)
    blocked: list[str] = field(default_factory=list)
    # Each failed attempt's error, per spec not yet completed, and the log of
    # the last one: its batch's.
    failures: dict[str, list[str]] = field(default_factory=dict)
    logs: dict[str, Path] = field(default_factory=dict)
    # When each spec first found missing each dependency this run sees built,
    # since it last got anywhere.
    unseen: dict[str, dict[str, float]] = field(default_factory=dict)
    # Attempts that found missing only specs this run sees built: their
    # worker's filesystem may not show those yet, so they rerun at the next
    # poll and not at once.
    waiting: list[list[str]] = field(default_factory=list)
    firsts: dict[tuple[Worker, str], Spec] = field(default_factory=dict)

    # A worker's filesystem can take about a minute to show a version another
    # node just published (NFS caches directory listings).
    unseen_grace_s: ClassVar[float] = 120.0

    def place(self, objs: Sequence[Spec]) -> list[Group]:
        """Say how ``objs`` run, and wait for the ones that are new to this run.

        Specs that run the same way are one group: the same class folder and
        process, and on each pool that can run them the same batch.
        """
        groups: dict[tuple[object, ...], Group] = {}
        nowhere = []
        for obj in objs:
            pools = tuple(
                (key, batch_of(obj, backend.worker, self.firsts))
                for key, backend in self.backends.items()
                if can_run(backend, obj)
            )
            if not pools:
                nowhere.append(obj._log_label)
            metadata = obj._metadata
            how = (
                obj._class_dir,
                pools,
                tuple(sorted(metadata.environment.items())),
                metadata.required_environment_variables,
                metadata.reuse,
            )
            if (group := groups.get(how)) is None:
                throttle = obj.throttle
                group = groups[how] = Group(
                    specs=[],
                    class_dir=obj._class_dir,
                    process=ProcessSettings.from_metadata(metadata),
                    pools={
                        key: Batch(batch=name, cap=cap) for key, (name, cap) in pools
                    },
                    throttle=None
                    if throttle is None
                    else JobThrottle(
                        key=obj._stored_name, max_running=throttle.max_running
                    ),
                )
            group.specs.append(obj.object_id)
        if nowhere:
            workers = ", ".join(
                str(backend.worker) for backend in self.backends.values()
            )
            raise RuntimeError(
                "no worker pool can run: "
                + ", ".join(sorted(nowhere))
                + f"; workers: {workers or 'none'}"
            )
        for obj in objs:
            if obj.object_id not in self.specs:
                self.specs[obj.object_id] = obj
                self.left.add(obj.object_id)
        return list(groups.values())

    def submit(
        self, groups: list[Group], lines: list[str], plan_path: Path
    ) -> SubmitMessage:
        """Ask the manager for the specs of ``groups``, which ``plan_path`` lists."""
        return SubmitMessage(
            pools={key: backend.pool() for key, backend in self.backends.items()},
            launch=self.launch.model_copy(update={"plan": plan_path}),
            lines=lines,
            groups=groups,
        )

    def drive(self, connection: ClientConnection, submit: SubmitMessage) -> None:
        """Send the launch and answer what comes back until every spec has ended."""
        connection.send(submit.model_dump_json())
        next_poll = time.monotonic() + _POLL_INTERVAL_S
        while self.left:
            try:
                raw = connection.recv(timeout=max(0.0, next_poll - time.monotonic()))
            except TimeoutError:
                raw = None
            replies: list[_Reply] = []
            if raw is not None:
                match manager_message_adapter.validate_json(raw):
                    case ResultMessage() as message:
                        replies = self.result(message)
                    case GaveUpMessage() as message:
                        self.gave_up(message)
                    case WarningMessage(reason=reason):
                        logger.warning("%s", reason)
                    case ErrorMessage(reason=reason):
                        self.fail(reason)
                    case _ as unreachable:
                        assert_never(unreachable)
            if time.monotonic() >= next_poll:
                replies += self.poll()
                next_poll = time.monotonic() + _POLL_INTERVAL_S
            for reply in replies:
                connection.send(reply.model_dump_json())
        self.finish()

    def poll(self) -> list[_Reply]:
        """Rerun the attempts that waited for their worker to see what is built."""
        waiting, self.waiting = self.waiting, []
        return [RerunMessage(specs=specs) for specs in waiting]

    def result(self, message: ResultMessage) -> list[_Reply]:
        """Apply how one create() call ended; say what its specs do next."""
        ids = [spec for spec in message.object_ids if spec in self.left]
        if not ids:
            logger.info("ignoring result for %s: not running", message.object_ids[0])
            return []
        # The log of that call: its batch's.
        ran = [self.specs[spec] for spec in message.object_ids if spec in self.specs]
        log = log_path_in(working_folder(ran))
        replies: list[_Reply]
        match message.result:
            case JobCompletedResult():
                for spec in ids:
                    self.left.remove(spec)
                    self.completed.add(spec)
                    # It is no longer retrying.
                    self.failures.pop(spec, None)
                    self.unseen.pop(spec, None)
                    logger.info(
                        "completed %s ok · %s",
                        self.specs[spec]._log_label,
                        format_duration(message.seconds),
                    )
                replies = []
            case JobFailedResult(error=error):
                replies = self._failed(ids, error, log)
            case JobBlockedResult() as blocked:
                for spec in ids:
                    self.logs[spec] = log
                    logger.info(
                        "blocked %s · %s",
                        self.specs[spec]._log_label,
                        missing_dependencies(len(blocked.dependencies)),
                    )
                try:
                    replies = self._blocked(ids, blocked)
                except Exception as exc:  # noqa: BLE001 -- runs user runs_on, accepts and batch fns
                    self.fail(f"{type(exc).__name__}: {exc}")
            case _ as unreachable:
                assert_never(unreachable)
        # A spec given up on counts as failed once the manager passes that on.
        if not any(isinstance(reply, GaveUpMessage) for reply in replies):
            self._progress()
        return replies

    def _failed(self, ids: Sequence[str], error: str, log: Path) -> list[_Reply]:
        """Count a failed attempt for each spec; rerun the ones with retries left."""
        rerun, gave_up = [], []
        for spec in ids:
            self.unseen.pop(spec, None)  # its earlier misses are behind it
            self.logs[spec] = log
            errors = self.failures.setdefault(spec, [])
            errors.append(error)
            will_retry = len(errors) <= self.max_retries_per_object
            (rerun if will_retry else gave_up).append(spec)
            # The traceback lives in the run's log; repeats say so instead.
            summary = error_summary(error)
            if len(errors) > 1 and error_summary(errors[-2]) == summary:
                summary = "same error"
            logger.log(
                logging.WARNING if will_retry else logging.ERROR,
                "failed %s · attempt %d/%d · %s",
                self.specs[spec]._log_label,
                len(errors),
                self.max_retries_per_object + 1,
                summary,
                extra={"path": log},
            )
        replies: list[_Reply] = []
        if gave_up:
            replies.append(self._give_up(gave_up, error))
        if rerun:
            replies.append(RerunMessage(specs=rerun))
        return replies

    def _blocked(self, ids: Sequence[str], blocked: JobBlockedResult) -> list[_Reply]:
        """What the specs of a create() that found specs missing do next.

        They run again once the missing ones are built, which are asked for
        here. Ones this run sees built may not be visible to the worker yet,
        so if the create() names only those it just reruns; once it has found
        one missing for over ``unseen_grace_s`` without getting anywhere in
        between, its specs are given up on.
        """
        lines = list(map(parse_json_object, blocked.lines))
        found = plan.Specs(lines)
        dependencies = [found[dependency] for dependency in blocked.dependencies]
        unbuilt = [
            dependency
            for dependency in dependencies
            if dependency.object_id not in self.completed and not dependency.done
        ]
        ended = [
            dependency
            for dependency in unbuilt
            if dependency.object_id in self.specs
            and dependency.object_id not in self.left
        ]
        if ended:
            why = f"needs {ended[0]._log_label}, which this run gave up on"
        else:
            # Each spec of the batch has its own clocks, which say the same.
            (why, *_) = [
                why_stuck(
                    self.unseen.setdefault(spec, {}),
                    blocked.dependencies,
                    bool(unbuilt),
                    self.unseen_grace_s,
                )
                for spec in ids
            ]
            if why is not None and dependencies:
                why += (
                    "; its worker may not see the storage it was written to, or "
                    "may be running different code than this run"
                )
        if why is not None:
            for spec in ids:
                self.failures.setdefault(spec, []).append(why)
                logger.error(
                    "failed %s · %s",
                    self.specs[spec]._log_label,
                    why,
                    extra={"path": self.logs[spec]},
                )
            return [self._give_up(ids, why)]
        if not unbuilt:
            self.waiting.append(list(ids))
            return []
        # The specs to run under the missing ones: new ones, and ones this
        # run still waits for, so that the new ones wait for them. Only ones
        # the worker found missing too, which have their lines.
        listed = {line["spec"] for line in lines if line.get("runs")}
        under = [
            obj
            for obj in missing(unbuilt)
            if obj.object_id in listed
            and (obj.object_id not in self.specs or obj.object_id in self.left)
        ]
        return [
            self.submit(self.place(under), blocked.lines, blocked.plan),
            RerunMessage(
                specs=list(ids), after=[dependency.object_id for dependency in unbuilt]
            ),
        ]

    def _give_up(self, ids: Sequence[str], error: str) -> GaveUpMessage:
        """Say that ``ids``, the specs of one create() call, will not run again."""
        return GaveUpMessage(
            specs=list(ids),
            error=error,
            log=self.logs[ids[0]],
            attempts={spec: len(self.failures[spec]) for spec in ids},
        )

    def gave_up(self, message: GaveUpMessage) -> None:
        """Specs that will not run again, and the ones that wait for them."""
        for spec in message.specs:
            if spec in self.left:
                self.left.remove(spec)
                self.failed.add(spec)
                if spec not in self.failures:  # another launch gave up on it
                    self.failures[spec] = [message.error] * message.attempts[spec]
                    self.logs[spec] = message.log
                    logger.error(
                        "failed %s · %s",
                        self.specs[spec]._log_label,
                        error_summary(message.error),
                        extra={"path": message.log},
                    )
        for spec in message.blocked:
            if spec in self.left:
                self.left.remove(spec)
                self.blocked.append(spec)
        self._progress()

    def _progress(self) -> None:
        parts = [f"progress {len(self.completed)}/{len(self.specs)}"]
        if self.blocked:
            parts.append(f"{len(self.blocked)} blocked")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if retrying := len(self.failures) - len(self.failed):
            parts.append(f"{retrying} retrying")
        logger.info(" · ".join(parts))

    def fail(self, message: str) -> NoReturn:
        logger.error(
            "run failed · %s", message, extra={"path": self.launch.execution_log}
        )
        raise RuntimeError(message)

    def finish(self) -> None:
        """Every spec has ended: succeed, or say what failed and why."""
        if not self.failed and not self.blocked:
            logger.info("run finished ok")
            return

        def plural(count: int) -> str:
            return f"{count} spec{'' if count == 1 else 's'}"

        # In order of first failure, like the run's log.
        failed = [
            (self.specs[spec], errors)
            for spec, errors in self.failures.items()
            if spec in self.failed
        ]
        lines = [f"{plural(len(failed))} failed, {len(self.blocked)} blocked"]
        for obj, errors in failed:
            lines.append(
                f"  {obj._log_label} · {len(errors)} attempt"
                f"{'' if len(errors) == 1 else 's'} · "
                f"{error_summary(errors[-1])} → "
                f"{_display_path(self.logs[obj.object_id])}"
            )
        lines += [f"  {self.specs[spec]._log_label} · blocked" for spec in self.blocked]

        headline = ["run failed"]
        if failed:
            headline.append(f"{plural(len(failed))} failed")
        if self.blocked:
            headline.append(f"{plural(len(self.blocked))} blocked")
        if len(failed) == 1:
            ((obj, errors),) = failed
            headline += [obj._log_label, error_summary(errors[-1])]
        logger.error(" · ".join(headline), extra={"path": self.launch.execution_log})
        raise RuntimeError("\n".join(lines))
