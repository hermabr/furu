"""The manager's queue: which specs go out next, and what scripts are told.

It holds no sockets and starts no threads. Scripts and workers are whatever
handles its caller passes, and what scripts are to be told collects in
``outbox``. The manager calls it under one lock.

It forms a batch when a worker asks: specs nobody has started, then stopped
batches that fit whole, then as much of the oldest stopped batch as there is
room for. A stopped batch is one that was handed out and did not finish. Its
working folder is named for its specs in order, so it is continued only when
it goes out alone.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from furu.dag import Dag
from furu.locking import is_active_lock
from furu.logging import _group_label, get_logger
from furu.protocol import plan_text
from furu.protocol.events import read_events
from furu.protocol.layout import (
    attempt_dir_in,
    attempts_dir_in,
    lock_path_in,
    run_name,
)
from furu.protocol.messages import (
    Batch,
    ErrorMessage,
    GaveUpMessage,
    Job,
    JobCompletedResult,
    JobResult,
    JobThrottle,
    Launch,
    ProcessSettings,
    RerunMessage,
    ResultMessage,
    SubmitMessage,
    WarningMessage,
)

logger = get_logger("manager")

# How long a stopped batch whose lock is held waits for the next look. An idle
# worker's handler looks this often, and checks that its worker is still there.
CHECK_INTERVAL_S = 5.0

type ToScript = ResultMessage | GaveUpMessage | WarningMessage | ErrorMessage


@dataclass(eq=False, slots=True)
class _Plan:
    """The lines of one plan file as they were sent, and what its jobs have in common."""

    launch: Launch
    # Each line under what other lines call it by: a spec's id, a value's
    # fingerprint.
    lines: dict[str, str] = field(default_factory=dict)


@dataclass(eq=False, slots=True)
class _Placed:
    """How the specs of one group run."""

    plan: _Plan
    class_dir: Path
    process: ProcessSettings
    pools: dict[str, Batch]
    throttle: JobThrottle | None


@dataclass(eq=False, slots=True)
class _Stopped:
    """A batch that was handed out and did not finish."""

    # In the order the opened line of its working folder lists them.
    specs: list[str]
    # The name of that folder.
    run: str
    # Whether the script that answers for it has yet to say that it runs again.
    held: bool
    # When its lock, found held, is next looked at.
    look_at: float = 0.0


@dataclass(eq=False, slots=True)
class _Lease:
    job: Job
    pool: str
    throttle: str | None
    started: float


@dataclass(eq=False)
class Queue[S: Hashable, W: Hashable]:
    """The specs that scripts ``S`` asked for, and the jobs workers ``W`` run."""

    # What every spec that is not finished waits for.
    dag: Dag = field(default_factory=Dag)
    # The plans each script sent lines of, by file: its launch's, and one for
    # each process whose create() asked for specs.
    plans: dict[S, dict[Path, _Plan]] = field(default_factory=dict)
    # The specs that a connected script asked for and that are not finished,
    # each under the plan that sent it first, and the scripts that asked. The
    # first of them answers for the spec.
    placed: dict[str, _Placed] = field(default_factory=dict)
    asked_by: dict[str, list[S]] = field(default_factory=dict)
    # The specs nobody has started that wait for nothing, in the order they
    # became ready: by pool, then batch name, then plan. Ordered dicts,
    # because a dict that is emptied from the front takes ever longer to
    # give its first key.
    unstarted: dict[str, OrderedDict[str, dict[_Plan, OrderedDict[str, None]]]] = field(
        default_factory=dict
    )
    # The stopped batches, oldest first, and the one each of their specs is in.
    stopped: list[_Stopped] = field(default_factory=list)
    stopped_in: dict[str, _Stopped] = field(default_factory=dict)
    # The specs a script gave up on. They stay unfinished, so what waits for
    # them does not run, until a script asks for them again.
    given_up: set[str] = field(default_factory=set)
    leases: dict[W, _Lease] = field(default_factory=dict)
    # Each class folder and pair of plans a smaller batch was warned about.
    warned: set[tuple[Path, frozenset[Path]]] = field(default_factory=set)
    # What to tell scripts, in order.
    outbox: list[tuple[S, ToScript]] = field(default_factory=list)

    # --- scripts ---------------------------------------------------------------

    def submit(self, script: S, message: SubmitMessage) -> int:
        """Hold a plan's lines and queue the specs of its groups that are new here.

        Returns how many were queued.
        """
        launch = message.launch
        plans = self.plans.setdefault(script, {})
        listed = plans.setdefault(launch.plan, _Plan(launch))
        values = set()
        for text in message.lines:
            line = json.loads(text)
            listed.lines[key := plan_text.key(line)] = text
            if "value" in line:
                values.add(key)
        holds = {key: plan_text.holds(listed.lines[key]) for key in values}
        new: dict[Path, list[str]] = {}
        for group in message.groups:
            placed = _Placed(
                listed, group.class_dir, group.process, group.pools, group.throttle
            )
            for spec in group.specs:
                holds[spec] = []
                if spec in self.given_up:
                    self.given_up.remove(spec)
                    self.placed[spec], self.asked_by[spec] = placed, [script]
                    self._queue(spec)
                elif spec in self.placed:
                    if script not in self.asked_by[spec]:
                        self.asked_by[spec].append(script)
                else:
                    self.placed[spec], self.asked_by[spec] = placed, [script]
                    # Otherwise it still runs for a script that left.
                    if spec not in self.dag.nodes:
                        holds[spec] = plan_text.holds(listed.lines[spec])
                        new.setdefault(group.class_dir, []).append(spec)
        self.dag.add(holds, values)
        for class_dir, specs in new.items():
            self._find_stopped(class_dir, specs)
        self._queue_ready()
        return sum(map(len, new.values()))

    def rerun(self, script: S, message: RerunMessage) -> None:
        theirs = [spec for spec in message.specs if self._answers(script, spec)]
        try:
            for spec in theirs:
                self.dag.block(spec, message.after)
        except RuntimeError as exc:  # a dependency cycle
            self.outbox.append((script, ErrorMessage(reason=str(exc))))
        for spec in theirs:
            self.stopped_in[spec].held = False
        self._queue_ready()

    def gave_up(self, script: S, message: GaveUpMessage) -> None:
        theirs = [spec for spec in message.specs if self._answers(script, spec)]
        told: dict[S, list[str]] = {}
        for spec in theirs:
            self._unstop(spec)
            self.given_up.add(spec)
            for asked in self.asked_by[spec]:
                told.setdefault(asked, []).append(spec)
        waiting = self.dag.waiting_for(theirs)
        for asked, specs in told.items():
            blocked = [spec for spec in waiting if asked in self.asked_by.get(spec, ())]
            reply = message.model_copy(update={"specs": specs, "blocked": blocked})
            self.outbox.append((asked, reply))
            # It counts them as ended, so they no longer run for it.
            for spec in blocked:
                self.asked_by[spec].remove(asked)
                if not self.asked_by[spec]:
                    self._forget(spec)
                    self.dag.complete(spec)
        self._queue_ready()

    def cancel(self, script: S) -> tuple[int, list[W]]:
        """Drop what ``script`` asked for.

        Returns how many queued specs were dropped, and the workers whose
        job now runs for nobody, to be stopped.
        """
        running = [
            worker
            for worker, lease in self.leases.items()
            if any(script in self.asked_by.get(spec, ()) for spec in lease.job.specs)
        ]
        dropped = self._drop(script)
        unwanted = set(self.unwanted())
        return dropped, [worker for worker in running if worker in unwanted]

    def leave(self, script: S) -> int:
        """Forget ``script`` and its plans; return how many queued specs were dropped."""
        dropped = self._drop(script)
        self.plans.pop(script, None)
        return dropped

    def unwanted(self) -> list[W]:
        """The workers whose job no script asks for any more."""
        return [
            worker
            for worker, lease in self.leases.items()
            if not any(spec in self.placed for spec in lease.job.specs)
        ]

    def _answers(self, script: S, spec: str) -> bool:
        """Whether ``spec`` waits for ``script`` to say what happens to it."""
        stopped = self.stopped_in.get(spec)
        return stopped is not None and stopped.held and self.asked_by[spec][0] is script

    def _drop(self, script: S) -> int:
        """Forget the specs only ``script`` asked for; return how many were not running.

        A running one finishes on its worker. What waited for a forgotten
        spec no longer does: whoever still wants that asks for the spec again
        when its create() finds it missing.
        """
        running = {spec for lease in self.leases.values() for spec in lease.job.specs}
        dropped = 0
        for spec, asked in list(self.asked_by.items()):
            if script not in asked:
                continue
            answered = asked[0] is script
            asked.remove(script)
            if not asked:
                self._forget(spec)
                if spec not in running:
                    self.dag.complete(spec)
                    dropped += 1
            elif answered and spec in self.stopped_in:
                self.stopped_in[spec].held = False
        self._queue_ready()
        return dropped

    def _find_stopped(self, class_dir: Path, specs: list[str]) -> None:
        """Take as stopped the working folders on disk that hold only ``specs``.

        They were left by workers this manager did not hand them to, or
        does not remember. One whose worker is still alive holds its lock.
        """
        try:
            runs = [
                name
                for name in os.listdir(attempts_dir_in(class_dir))
                if "." not in name
            ]
        except FileNotFoundError:
            return
        by_hash = {spec.rpartition(":")[2]: spec for spec in specs}
        for run in runs:
            hashes = [
                spec
                for opened in read_events(attempt_dir_in(class_dir, run))[:1]
                for spec in opened.get("specs", ())
            ]
            if hashes and all(
                spec in by_hash and by_hash[spec] not in self.stopped_in
                for spec in hashes
            ):
                self._stop([by_hash[spec] for spec in hashes], run, held=False)

    def _stop(self, specs: list[str], run: str, *, held: bool) -> None:
        stopped = _Stopped(specs, run, held)
        self.stopped.append(stopped)
        for spec in specs:
            self.stopped_in[spec] = stopped

    def _queue(self, spec: str) -> None:
        placed = self.placed[spec]
        for pool, batch in placed.pools.items():
            batches = self.unstarted.setdefault(pool, OrderedDict())
            plans = batches.setdefault(batch.batch, {})
            plans.setdefault(placed.plan, OrderedDict())[spec] = None

    def _queue_ready(self) -> None:
        """Queue the specs that have nothing left to wait for.

        A spec of a stopped batch goes out with that batch.
        """
        for spec in self.dag.ready:
            if spec not in self.stopped_in:
                self._queue(spec)
        self.dag.ready.clear()

    def _unqueue(self, spec: str) -> None:
        placed = self.placed[spec]
        for pool, batch in placed.pools.items():
            batches = self.unstarted.get(pool, {})
            plans = batches.get(batch.batch, {})
            queue = plans.get(placed.plan, {})
            queue.pop(spec, None)
            if not queue:
                plans.pop(placed.plan, None)
            if not plans:
                batches.pop(batch.batch, None)

    def _unstop(self, spec: str) -> None:
        if (stopped := self.stopped_in.pop(spec, None)) is not None:
            stopped.specs.remove(spec)
            if not stopped.specs:
                self.stopped.remove(stopped)

    def _forget(self, spec: str) -> None:
        self._unqueue(spec)
        self._unstop(spec)
        self.given_up.discard(spec)
        del self.placed[spec], self.asked_by[spec]

    # --- workers ---------------------------------------------------------------

    def hand_out(self, worker: W, pool: str) -> Job | None:
        """Give a worker of ``pool`` its next job, if there is one for it.

        Unstarted specs go first, in the order they became ready. A stopped
        batch leads a job only when no unstarted spec can.
        """
        throttled = self._throttled()
        lead = next(
            (
                placed
                for plans in self.unstarted.get(pool, {}).values()
                for queue in plans.values()
                if not throttled(placed := self.placed[next(iter(queue))])
            ),
            None,
        )
        unstarted: list[str] = []
        if lead is not None:
            batch = lead.pools[pool]
            queue = self.unstarted[pool][batch.batch][lead.plan]
            unstarted = list(islice(queue, batch.cap))
        else:
            stopped = next(
                self._startable(
                    lambda spec: (
                        pool in self.placed[spec].pools
                        and not throttled(self.placed[spec])
                    )
                ),
                None,
            )
            if stopped is None:
                return None
            lead = self.placed[stopped.specs[0]]
        specs = self._fill(unstarted, pool, lead)
        if not specs:  # the stopped batch's lock was taken since
            return None
        throttle = self.placed[specs[0]].throttle
        job = self._job(specs)
        self.leases[worker] = _Lease(
            job, pool, None if throttle is None else throttle.key, time.monotonic()
        )
        return job

    def finish(self, worker: W, result: JobResult) -> None:
        """Apply how ``worker``'s job ended, and tell the scripts that asked for its specs.

        Specs that did not complete are a stopped batch for each script that
        answers for some, held until it says whether they run again.
        """
        lease = self.leases.pop(worker)
        theirs = []
        for spec in lease.job.specs:
            if spec in self.placed:
                theirs.append(spec)
            elif spec in self.dag.nodes:  # every script that asked for it left
                self.dag.complete(spec)
        told: dict[S, list[str]] = {}
        if isinstance(result, JobCompletedResult):
            for spec in theirs:
                for asked in self.asked_by[spec]:
                    told.setdefault(asked, []).append(spec)
                self._forget(spec)
                self.dag.complete(spec)
        elif theirs:
            for spec in theirs:
                self._unstop(spec)  # found on disk when its script came back
                told.setdefault(self.asked_by[spec][0], []).append(spec)
            run = run_name([spec.rpartition(":")[2] for spec in lease.job.specs])
            for specs in told.values():
                self._stop(specs, run, held=True)
        self._queue_ready()
        seconds = time.monotonic() - lease.started
        for script, specs in told.items():
            message = ResultMessage(object_ids=specs, result=result, seconds=seconds)
            self.outbox.append((script, message))

    def _job(self, specs: list[str]) -> Job:
        """The job that runs ``specs``: their lines and the lines their fields reach."""
        placed = self.placed[specs[0]]
        lines = placed.plan.lines
        reached = dict.fromkeys(specs)
        pending = list(specs)
        while pending:
            for key in plan_text.holds(lines[pending.pop()], returned=False):
                # What a serializer wrote can look like a pointer and have no line.
                if key not in reached and key in lines:
                    reached[key] = None
                    pending.append(key)
        if "moved" in lines:
            reached["moved"] = None
        return Job(
            specs=specs,
            lines=[lines[key] for key in reached],
            class_dir=placed.class_dir,
            process=placed.process,
            **dict(placed.plan.launch),
        )

    def _throttled(self) -> Callable[[_Placed], bool]:
        """Whether a spec's class already runs in as many jobs as its throttle allows.

        Throttles count jobs across all scripts, so a running batch counts once.
        """
        running = Counter(
            lease.throttle for lease in self.leases.values() if lease.throttle
        )

        def throttled(placed: _Placed) -> bool:
            throttle = placed.throttle
            return (
                throttle is not None and running[throttle.key] >= throttle.max_running
            )

        return throttled

    def _startable(self, wanted: Callable[[str], bool]) -> Iterator[_Stopped]:
        """The stopped batches that can go out now, oldest first.

        ``wanted`` says whether the caller has a use for a batch, given its
        first spec. A batch whose working folder is locked is still being
        computed, or its worker died less than a lock's lifetime ago.
        """
        now = time.monotonic()
        for stopped in list(self.stopped):
            if (
                stopped.held
                or now < stopped.look_at
                or any(map(self.dag.waits, stopped.specs))
                or not wanted(stopped.specs[0])
            ):
                continue
            class_dir = self.placed[stopped.specs[0]].class_dir
            if is_active_lock(lock_path_in(class_dir, stopped.run)):
                stopped.look_at = now + CHECK_INTERVAL_S
            else:
                yield stopped

    def _fill(self, specs: list[str], pool: str, placed: _Placed) -> list[str]:
        """Make ``specs``, unstarted, the job of a worker of ``pool``, filled from stopped batches.

        Those that fit whole go in first, then as much of the oldest as there
        is room for. The rest of a batch that was cut are unstarted specs
        again. A stopped batch that is not the whole job starts over in a new
        working folder, and its script is told where the old one is.
        """
        listed, batch = placed.plan, placed.pools[pool]

        def joins(spec: str) -> bool:
            other = self.placed[spec]
            return other.plan is listed and other.pools.get(pool) == batch

        for spec in specs:
            self._unqueue(spec)
        used: list[_Stopped] = []
        if len(specs) < batch.cap:
            startable = list(self._startable(joins))
            for stopped in startable:
                if len(specs) + len(stopped.specs) <= batch.cap and all(
                    map(joins, stopped.specs)
                ):
                    specs += stopped.specs
                    used.append(stopped)
            for stopped in startable:
                if len(specs) < batch.cap and stopped not in used:
                    room = batch.cap - len(specs)
                    specs += [spec for spec in stopped.specs if joins(spec)][:room]
                    used.append(stopped)
        taken = set(specs)
        for stopped in used:
            self.stopped.remove(stopped)
            for spec in stopped.specs:
                del self.stopped_in[spec]
                if spec not in taken:
                    self._queue(spec)

        if not specs:
            return specs
        script = self.asked_by[specs[0]][0]
        label = _group_label(specs[0], len(specs))
        run = run_name([spec.rpartition(":")[2] for spec in specs])
        for stopped in used:
            if stopped.run != run:
                reason = (
                    f"{label} starts in a new working folder: it holds specs of a "
                    "batch that stopped and is not that batch alone, so what the "
                    "batch had done is dropped. Its working folder stays in "
                    f"attempts/: {attempt_dir_in(placed.class_dir, stopped.run)}"
                )
                logger.warning("%s", reason)
                self.outbox.append((script, WarningMessage(reason=reason)))
        if len(specs) < batch.cap:
            for other in self.unstarted.get(pool, {}).get(batch.batch, {}):
                plans = frozenset({listed.launch.plan, other.launch.plan})
                if other is not listed and (placed.class_dir, plans) not in self.warned:
                    self.warned.add((placed.class_dir, plans))
                    reason = (
                        f"{label} is a smaller batch than its worker can take "
                        f"({batch.cap}), while unstarted specs with the same batch "
                        "key wait: a batch holds specs from one plan, and those "
                        f"are listed in {other.launch.plan}, not "
                        f"{listed.launch.plan}"
                    )
                    logger.warning("%s", reason)
                    self.outbox.append((script, WarningMessage(reason=reason)))
        return specs

    def demand(self, pool: str, max_workers: int) -> int:
        """Workers ``pool`` should have: busy ones plus jobs it could start, up to ``max_workers``."""
        busy = sum(1 for lease in self.leases.values() if lease.pool == pool)
        running = Counter(
            lease.throttle for lease in self.leases.values() if lease.throttle
        )
        startable = sum(
            not stopped.held
            and pool in self.placed[stopped.specs[0]].pools
            and not any(map(self.dag.waits, stopped.specs))
            for stopped in self.stopped
        )
        for plans in self.unstarted.get(pool, {}).values():
            if busy + startable >= max_workers:
                break
            for queue in plans.values():
                placed = self.placed[next(iter(queue))]
                jobs = -(-len(queue) // placed.pools[pool].cap)
                if (throttle := placed.throttle) is not None:
                    free = throttle.max_running - running[throttle.key]
                    jobs = max(0, min(jobs, free))
                    running[throttle.key] += jobs
                startable += jobs
        return min(busy + startable, max_workers)
