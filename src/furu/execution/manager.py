"""The furu manager: one queue and one set of worker pools for every launch.

It holds only memory. A script sends its plan's lines and its pool
definitions on every connection, so a restarted manager needs no state file:
scripts plan again from disk and resend, and the working folders in
``attempts/`` say which batches were started and who holds each. Run it with
``uvx furu@<version> manager``.

A worker is one allocation whose ``slots`` its jobs share. One thread hands
out jobs as slots free up, each to the worker it fits most tightly. It forms a
batch from specs nobody has started, then stopped batches that fit whole, then
as much of the oldest stopped batch as there is room for. A stopped batch is
one that was handed out and did not finish. Its working folder is named for
its specs in order, so it is continued only when it goes out alone.
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from itertools import count, islice
from pathlib import Path
from secrets import token_urlsafe
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State
from websockets.sync.server import ServerConnection, basic_auth, serve

from furu import plan
from furu.dag import Dag
from furu.logging import _group_label, get_logger
from furu.storage import attempt_lock
from furu.storage._layout import (
    attempt_parent_in,
    attempts_dir_in,
    list_names,
    run_name,
)
from furu.storage.events import read_events
from furu.worker.backends.local import LocalThreadPool
from furu.worker.backends.protocol import WorkerPool
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.messages import (
    Batch,
    CancelJobMessage,
    CancelMessage,
    ErrorMessage,
    GaveUpMessage,
    HelloMessage,
    Job,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    JobResultMessage,
    JobThrottle,
    Launch,
    LocalPool,
    Pool,
    ProcessSettings,
    RerunMessage,
    ResultMessage,
    ScriptMessage,
    SlurmPool,
    SubmitMessage,
    WarningMessage,
    first_message_adapter,
    script_message_adapter,
)

logger = get_logger("manager")

# How often jobs are handed out and idle workers looked at when nothing changed.
_IDLE_CHECK_INTERVAL_S = 5.0
# How often the working folder of a stopped batch is looked at again.
_HELD_CHECK_INTERVAL_S = 30.0

type _ToScript = ResultMessage | GaveUpMessage | WarningMessage | ErrorMessage


@dataclass(eq=False, slots=True)
class _Plan:
    """The lines of one plan file as they were sent, and what its jobs have in common."""

    launch: Launch
    # Each line under what other lines call it by: a spec's id, a value's
    # fingerprint.
    lines: dict[str, str] = field(default_factory=dict)


@dataclass(eq=False)
class _Script:
    connection: ServerConnection
    pools: dict[str, Pool] = field(default_factory=dict)
    # The plans it sent lines of, by file: its launch's, and one for each
    # process whose create() asked for specs.
    plans: dict[Path, _Plan] = field(default_factory=dict)

    def send(self, message: _ToScript) -> None:
        with suppress(ConnectionClosed):
            self.connection.send(message.model_dump_json())


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
    # When its working folder is next checked, and what the last check found:
    # None if free to take, "" until first checked, else the held reason warned.
    look_at: float = 0.0
    why: str | None = ""


@dataclass(eq=False, slots=True)
class _Allocation:
    """A connected worker, whose jobs share its ``slots``."""

    connection: ServerConnection
    pool: str
    slots: int
    name: str
    used: int = 0
    # Since when it has had no job.
    idle_since: float = field(default_factory=time.monotonic)

    @property
    def free(self) -> int:
        return self.slots - self.used


@dataclass(eq=False, slots=True)
class _Lease:
    specs: list[str]
    job: Job
    pool: str
    throttle: str | None
    started: float
    allocation: _Allocation


@dataclass(eq=False)
class Manager:
    url: str
    manager_dir: Path | None  # None for a script's in-process manager
    lock: threading.Condition = field(default_factory=threading.Condition)
    scripts: list[_Script] = field(default_factory=list)
    pools: dict[str, WorkerPool] = field(default_factory=dict)
    stopping: bool = False
    # What every spec that is not finished waits for.
    dag: Dag = field(default_factory=Dag)
    # The specs that a connected script asked for and that are not finished,
    # each under the plan that sent it first, and the scripts that asked. The
    # first of them answers for the spec.
    placed: dict[str, _Placed] = field(default_factory=dict)
    asked_by: dict[str, list[_Script]] = field(default_factory=dict)
    # The specs nobody has started that wait for nothing, in the order they
    # became ready: by pool, then batch, then plan. Ordered dicts,
    # because a dict that is emptied from the front takes ever longer to
    # give its first key.
    unstarted: dict[str, OrderedDict[Batch, dict[_Plan, OrderedDict[str, None]]]] = (
        field(default_factory=dict)
    )
    # The stopped batches, oldest first, and the one each of their specs is in.
    stopped: list[_Stopped] = field(default_factory=list)
    stopped_in: dict[str, _Stopped] = field(default_factory=dict)
    # The specs a script gave up on. They stay unfinished, so what waits for
    # them does not run, until a script asks for them again.
    gave_up: set[str] = field(default_factory=set)
    # The connected workers, oldest first, and the jobs they run, by lease.
    allocations: list[_Allocation] = field(default_factory=list)
    leases: dict[int, _Lease] = field(default_factory=dict)
    lease_ids: Iterator[int] = field(default_factory=count)
    # What the script that last asked for each pool asked of its workers.
    last_asked: dict[str, Pool] = field(default_factory=dict)
    # Each class folder and pair of plans a smaller batch was warned about.
    warned: set[tuple[Path, frozenset[Path]]] = field(default_factory=set)
    # What to send once the lock is released: a send can wait on its reader.
    outbox: list[tuple[_Script, _ToScript]] = field(default_factory=list)

    @classmethod
    @contextmanager
    def serve(
        cls, *, listen_host: str, connect_host: str, manager_dir: Path | None
    ) -> Generator[Manager]:
        token = token_urlsafe(32)
        server = serve(
            lambda connection: manager._serve(connection),
            listen_host,
            0,
            process_request=basic_auth(credentials=("furu", token)),
            max_size=None,
        )
        port = server.socket.getsockname()[1]
        manager = cls(
            url=f"ws://furu:{token}@{connect_host}:{port}", manager_dir=manager_dir
        )
        thread = threading.Thread(target=server.serve_forever, name="furu-manager")
        thread.start()
        dispatch = threading.Thread(target=manager._dispatch, name="furu-dispatch")
        dispatch.start()
        try:
            yield manager
        finally:
            with manager.lock:
                # A script that just left may still be forwarding its cancel.
                manager.lock.wait_for(
                    lambda: all(
                        s.connection.state is State.OPEN for s in manager.scripts
                    ),
                    timeout=5,
                )
                manager.stopping = True
                manager.lock.notify_all()
            dispatch.join(timeout=10)
            server.shutdown()  # closes every connection and waits for handlers
            thread.join(timeout=10)
            for pool in manager.pools.values():
                pool.stop()

    def _serve(self, connection: ServerConnection) -> None:
        try:
            match first_message_adapter.validate_json(connection.recv(timeout=10)):
                case SubmitMessage() as submit:
                    self._serve_script(connection, submit)
                case HelloMessage() as hello:
                    self._serve_worker(connection, hello)
                case _ as unreachable:
                    assert_never(unreachable)
        except (ConnectionClosed, TimeoutError):
            pass

    def _flush(self) -> None:
        with self.lock:
            outbox, self.outbox = self.outbox, []
        for script, message in outbox:
            script.send(message)

    # --- scripts ---------------------------------------------------------------

    def _serve_script(
        self, connection: ServerConnection, submit: SubmitMessage
    ) -> None:
        script = _Script(connection)
        with self.lock:
            self.scripts.append(script)
        logger.info("script connected · %s", _peer(connection))
        message: ScriptMessage = submit
        try:
            while True:
                match message:
                    case SubmitMessage():
                        self._submit(script, message)
                    case RerunMessage():
                        self._rerun(script, message)
                    case GaveUpMessage():
                        self._gave_up(script, message)
                    case CancelMessage():
                        self._cancel(script)
                    case _ as unreachable:
                        assert_never(unreachable)
                self._flush()
                message = script_message_adapter.validate_json(connection.recv())
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self.scripts.remove(script)
                dropped = self._drop(script)
                running = sum(
                    not any(spec in self.placed for spec in lease.specs)
                    for lease in self.leases.values()
                )
                self.lock.notify_all()
            logger.info(
                "script left · %s · dropped %d queued · %d still running",
                _peer(connection),
                dropped,
                running,
            )

    def _submit(self, script: _Script, message: SubmitMessage) -> None:
        with self.lock:
            script.pools = message.pools
            self.last_asked |= message.pools
            for key, definition in message.pools.items():
                if key not in self.pools:
                    self.pools[key] = self._start_pool(key, definition)
            queued = self._add(script, message)
            self.lock.notify_all()
        if queued:
            logger.info("queued %d specs from %s", queued, _peer(script.connection))

    def _add(self, script: _Script, message: SubmitMessage) -> int:
        """Hold a plan's lines and queue the specs of its groups that are new here."""
        launch = message.launch
        listed = script.plans.setdefault(launch.plan, _Plan(launch))
        values = set()
        for text in message.lines:
            line = json.loads(text)
            listed.lines[key := plan.key(line)] = text
            if "value" in line:
                values.add(key)
        holds = {key: plan.holds(listed.lines[key]) for key in values}
        new: dict[Path, list[str]] = {}
        for group in message.groups:
            placed = _Placed(
                listed, group.class_dir, group.process, group.pools, group.throttle
            )
            for spec in group.specs:
                holds[spec] = []
                if spec in self.gave_up:
                    self.gave_up.remove(spec)
                    self.placed[spec], self.asked_by[spec] = placed, [script]
                    self._queue(spec)
                elif spec in self.placed:
                    if script not in self.asked_by[spec]:
                        self.asked_by[spec].append(script)
                else:
                    self.placed[spec], self.asked_by[spec] = placed, [script]
                    # Otherwise it still runs for a script that left.
                    if spec not in self.dag.nodes:
                        holds[spec] = plan.holds(listed.lines[spec])
                        new.setdefault(group.class_dir, []).append(spec)
        self.dag.add(holds, values)
        for class_dir, specs in new.items():
            self._find_stopped(class_dir, specs)
        self._queue_ready()
        return sum(map(len, new.values()))

    def _find_stopped(self, class_dir: Path, specs: list[str]) -> None:
        """Take as stopped the working folders on disk that hold only ``specs``,
        none of which waits for another.

        They were left by workers this manager did not hand them to, or
        does not remember. One whose worker is still alive is that worker's
        by its name.
        """
        runs = [
            name for name in list_names(attempts_dir_in(class_dir)) if "." not in name
        ]
        by_hash = {spec.rpartition(":")[2]: spec for spec in specs}
        for run in runs:
            attempt = attempt_lock.current(attempt_parent_in(class_dir, run))
            hashes = [
                spec
                for opened in (read_events(attempt)[:1] if attempt else ())
                for spec in opened.get("specs", ())
            ]
            if hashes and all(
                spec in by_hash and by_hash[spec] not in self.stopped_in
                for spec in hashes
            ):
                batch = [by_hash[spec] for spec in hashes]
                # One that waits for another could never go out with it.
                if not any(
                    spec in batch
                    for one in batch
                    for spec in self.dag.waiting_for([one])
                ):
                    self._stop(batch, run, held=False)

    def _stop(self, specs: list[str], run: str, *, held: bool) -> None:
        stopped = _Stopped(specs, run, held)
        self.stopped.append(stopped)
        for spec in specs:
            self.stopped_in[spec] = stopped

    def _queue(self, spec: str) -> None:
        placed = self.placed[spec]
        for pool, batch in placed.pools.items():
            batches = self.unstarted.setdefault(pool, OrderedDict())
            plans = batches.setdefault(batch, {})
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
            plans = batches.get(batch, {})
            queue = plans.get(placed.plan, {})
            queue.pop(spec, None)
            if not queue:
                plans.pop(placed.plan, None)
            if not plans:
                batches.pop(batch, None)

    def _unstop(self, spec: str) -> None:
        if (stopped := self.stopped_in.pop(spec, None)) is not None:
            stopped.specs.remove(spec)
            if not stopped.specs:
                self.stopped.remove(stopped)

    def _forget(self, spec: str) -> None:
        self._unqueue(spec)
        self._unstop(spec)
        self.gave_up.discard(spec)
        del self.placed[spec], self.asked_by[spec]

    def _start_pool(self, key: str, definition: Pool) -> WorkerPool:
        logger.info("starting pool %s", key)
        match definition:
            case SlurmPool():
                if self.manager_dir is None:
                    raise RuntimeError("slurm pools need a shared furu manager")
                return SlurmWorkerPool(
                    manager=self,
                    key=key,
                    definition=definition,
                    directory=self.manager_dir / "workers" / key.replace(":", "-"),
                )
            case LocalPool():
                return LocalThreadPool(manager=self, key=key, definition=definition)
            case _ as unreachable:
                assert_never(unreachable)

    def _rerun(self, script: _Script, message: RerunMessage) -> None:
        with self.lock:
            theirs = [spec for spec in message.specs if self._answers(script, spec)]
            try:
                for spec in theirs:
                    self.dag.block(spec, message.after)
            except RuntimeError as exc:  # a dependency cycle
                self.outbox.append((script, ErrorMessage(reason=str(exc))))
            for spec in theirs:
                self.stopped_in[spec].held = False
            self._queue_ready()
            self.lock.notify_all()

    def _gave_up(self, script: _Script, message: GaveUpMessage) -> None:
        with self.lock:
            theirs = [spec for spec in message.specs if self._answers(script, spec)]
            told: dict[_Script, list[str]] = {}
            for spec in theirs:
                self._unstop(spec)
                self.gave_up.add(spec)
                for asked in self.asked_by[spec]:
                    told.setdefault(asked, []).append(spec)
            waiting = self.dag.waiting_for(theirs)
            for asked, specs in told.items():
                blocked = [
                    spec for spec in waiting if asked in self.asked_by.get(spec, ())
                ]
                reply = message.model_copy(update={"specs": specs, "blocked": blocked})
                self.outbox.append((asked, reply))
                # It counts them as ended, so they no longer run for it.
                for spec in blocked:
                    self.asked_by[spec].remove(asked)
                    if not self.asked_by[spec]:
                        self._forget(spec)
                        self.dag.complete(spec)
            self._queue_ready()
            self.lock.notify_all()

    def _answers(self, script: _Script, spec: str) -> bool:
        """Whether ``spec`` waits for ``script`` to say what happens to it."""
        stopped = self.stopped_in.get(spec)
        return stopped is not None and stopped.held and self.asked_by[spec][0] is script

    def _cancel(self, script: _Script) -> None:
        with self.lock:
            running = [
                lease
                for lease in self.leases.values()
                if any(script in self.asked_by.get(spec, ()) for spec in lease.specs)
            ]
            dropped = self._drop(script)
            # Not the ones another script still asks for.
            cancelled = [
                lease
                for lease in running
                if not any(spec in self.placed for spec in lease.specs)
            ]
            self.lock.notify_all()
        logger.info(
            "cancelled %d queued and %d running jobs from %s",
            dropped,
            len(cancelled),
            _peer(script.connection),
        )
        for lease in cancelled:
            with suppress(ConnectionClosed):
                lease.allocation.connection.send(
                    CancelJobMessage(lease=lease.job.lease).model_dump_json()
                )

    def _drop(self, script: _Script) -> int:
        """Forget the specs only ``script`` asked for; return how many were not running.

        A running one finishes on its worker. What waited for a forgotten
        spec no longer does: whoever still wants that asks for the spec again
        when its create() finds it missing.
        """
        running = {spec for lease in self.leases.values() for spec in lease.specs}
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

    def notify(self, pool: str, message: WarningMessage | ErrorMessage) -> None:
        """Tell every script that uses ``pool``, e.g. that its workers fail."""
        (logger.warning if isinstance(message, WarningMessage) else logger.error)(
            "%s · %s", pool, message.reason
        )
        with self.lock:
            scripts = [script for script in self.scripts if pool in script.pools]
        for script in scripts:
            script.send(message)

    # --- workers ---------------------------------------------------------------

    def _serve_worker(self, connection: ServerConnection, hello: HelloMessage) -> None:
        """Take a worker's results until it leaves; jobs go out from ``_dispatch``."""
        worker = hello.worker
        with self.lock:
            known = hello.pool in self.pools
            if known:
                allocation = _Allocation(connection, hello.pool, hello.slots, worker)
                self.allocations.append(allocation)
                self.lock.notify_all()
        if not known:
            logger.warning("worker %s belongs to no pool here; closing", worker)
            return
        logger.debug("worker connected · %s", worker)
        try:
            while True:
                ended = JobResultMessage.model_validate_json(connection.recv())
                with self.lock:
                    lease = self.leases.pop(ended.lease)
                    if isinstance(ended.result, JobCompletedResult):
                        self.pools[lease.pool].job_completed()
                    self._end(lease, ended.result)
                    self.lock.notify_all()
                self._flush()
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                with suppress(ValueError):  # gone already if it idled too long
                    self.allocations.remove(allocation)
                lost = [
                    lease
                    for lease in self.leases.values()
                    if lease.allocation is allocation
                ]
                for lease in lost:
                    del self.leases[lease.job.lease]
                    # Once stopping, the worker finishes its jobs alone; they
                    # haven't failed.
                    if not self.stopping:
                        error = f"worker {worker} disconnected"
                        self._end(lease, JobFailedResult(error=error))
                self.lock.notify_all()
            if lost and not self.stopping:
                logger.warning("lost %s mid-job", worker, extra={"path": hello.log})
            self._flush()

    def _end(self, lease: _Lease, result: JobResult) -> None:
        """Free a lease's slots and apply how its job ended."""
        allocation = lease.allocation
        allocation.used -= lease.job.slots
        if not allocation.used:
            allocation.idle_since = time.monotonic()
        self._finish(lease, result)

    def _dispatch(self) -> None:
        """Hand out jobs as slots free up, and let go of workers idle too long.

        A worker let go of has its connection closed, so it exits.
        """
        while True:
            with self.lock:
                pools = {a.pool for a in self.allocations if a.free}
            # Looked at first: a job is filled from the stopped batches whose
            # working folders are free.
            if [pool for pool in sorted(pools) if self._check_stopped(pool)]:
                self._flush()  # a check can leave a warning
            with self.lock:
                if self.stopping:
                    return
                leases = [
                    lease
                    for pool in sorted({a.pool for a in self.allocations})
                    for lease in self._hand_out(pool)
                ]
                idle = [a for a in self.allocations if not a.used]
                left = {a: self._idle_left(a) for a in idle}
                done = [a for a in idle if left[a] <= 0]
                for allocation in done:
                    self.allocations.remove(allocation)
                if not leases and not done:
                    waits = [wait for wait in left.values() if wait > 0]
                    self.lock.wait(timeout=min([_IDLE_CHECK_INTERVAL_S, *waits]))
            for lease in leases:
                job, allocation = lease.job, lease.allocation
                where = allocation.name
                if allocation.slots > 1:
                    where += f" ({job.slots}/{allocation.slots} slots)"
                logger.info(
                    "leased %s to %s", job.label, where, extra={"path": job.folder}
                )
                # One that is gone fails the job when its handler sees it go.
                with suppress(ConnectionClosed):
                    allocation.connection.send(job.model_dump_json())
            for allocation in done:
                # Not here: closing waits for the worker to answer.
                threading.Thread(
                    target=allocation.connection.close, name="furu-close-idle"
                ).start()
            self._flush()  # forming a batch can leave a warning

    def _finish(self, lease: _Lease, result: JobResult) -> None:
        """Apply how a job ended, and tell the scripts that asked for its specs.

        Specs that did not complete are a stopped batch for each script that
        answers for some, held until it says whether they run again.
        """
        theirs = []
        for spec in lease.specs:
            if spec in self.placed:
                theirs.append(spec)
            elif spec in self.dag.nodes:  # every script that asked for it left
                self.dag.complete(spec)
        told: dict[_Script, list[str]] = {}
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
            run = run_name([spec.rpartition(":")[2] for spec in lease.specs])
            for specs in told.values():
                self._stop(specs, run, held=True)
        self._queue_ready()
        seconds = time.monotonic() - lease.started
        for script, specs in told.items():
            message = ResultMessage(object_ids=specs, result=result, seconds=seconds)
            self.outbox.append((script, message))

    def _check_stopped(self, pool: str) -> bool:
        """Check whether due stopped batches' working folders are free.

        Asked without the lock: asking about a Slurm job's folder runs Slurm
        commands. The script that answers for a batch whose folder is
        another's is told why, once for each reason.
        """
        with self.lock:
            now = time.monotonic()
            due = []
            for stopped in self.stopped:
                placed = self.placed[stopped.specs[0]]
                if (
                    not stopped.held
                    and now >= stopped.look_at
                    and pool in placed.pools
                    and not any(map(self.dag.waits, stopped.specs))
                ):
                    stopped.look_at = now + _HELD_CHECK_INTERVAL_S
                    due.append(
                        (stopped, attempt_parent_in(placed.class_dir, stopped.run))
                    )
        if not due:
            return False
        whys: list[str | None] = []
        for _, folder in due:
            try:
                attempt_lock.free(folder)
            except (attempt_lock.HeldError, OSError, subprocess.SubprocessError) as exc:
                whys.append(str(exc))
            else:
                whys.append(None)
        next_look = time.monotonic() + _HELD_CHECK_INTERVAL_S
        with self.lock:
            for (stopped, _), why in zip(due, whys):
                stopped.look_at = next_look
                # Not if its specs finished or were given up on meanwhile.
                if why is not None and why != stopped.why and stopped in self.stopped:
                    label = _group_label(stopped.specs[0], len(stopped.specs))
                    reason = f"{label} waits: {why}"
                    logger.warning("%s", reason)
                    script = self.asked_by[stopped.specs[0]][0]
                    self.outbox.append((script, WarningMessage(reason=reason)))
                stopped.why = why
            self.lock.notify_all()
        return True

    def _hand_out(self, pool: str) -> list[_Lease]:
        """Give ``pool``'s workers the jobs that fit in their free slots.

        Jobs go in the order of the queue: unstarted specs in the order they
        became ready, then stopped batches. Each goes to the worker with the
        fewest free slots that hold it, so wide gaps stay for wide jobs. The
        first job that fits nowhere holds back the worker with the most free
        slots, the oldest of a tie: nothing goes there until that job fits, so
        narrower jobs cannot starve it.
        """
        allocations = [
            a
            for a in self.allocations
            if a.pool == pool and a.connection.state is State.OPEN
        ]
        leases: list[_Lease] = []
        reserved: _Allocation | None = None
        while any(a.free for a in allocations if a is not reserved):
            for lead, queue in self._leads(pool, self._throttled()):
                slots = lead.pools[pool].slots
                fits = [a for a in allocations if a is not reserved and a.free >= slots]
                if fits:
                    break
                if reserved is None:
                    reserved = max(allocations, key=lambda a: a.free)
            else:
                break
            allocation = min(fits, key=lambda a: a.free)
            unstarted = (
                [] if queue is None else list(islice(queue, lead.pools[pool].cap))
            )
            specs = self._fill(unstarted, pool, lead)
            throttle = self.placed[specs[0]].throttle
            lease = _Lease(
                specs,
                self._job(specs, pool, lease=next(self.lease_ids), slots=slots),
                pool,
                None if throttle is None else throttle.key,
                time.monotonic(),
                allocation,
            )
            self.leases[lease.job.lease] = lease
            allocation.used += slots
            leases.append(lease)
        return leases

    def _leads(
        self, pool: str, throttled: Callable[[_Placed], bool]
    ) -> Iterator[tuple[_Placed, OrderedDict[str, None] | None]]:
        """What can lead a job of ``pool`` now, in the order they go out.

        Each unstarted queue, with that queue; then each stopped batch, which
        leads a job only when no unstarted spec can.
        """
        for plans in self.unstarted.get(pool, {}).values():
            for queue in plans.values():
                if not throttled(placed := self.placed[next(iter(queue))]):
                    yield placed, queue
        for stopped in self._startable(
            lambda spec: (
                pool in self.placed[spec].pools and not throttled(self.placed[spec])
            )
        ):
            yield self.placed[stopped.specs[0]], None

    def _idle_left(self, allocation: _Allocation) -> float:
        """How many seconds an idle worker still waits for a job.

        Busy workers of its pool rank first, then idle ones, the longest idle
        last. The nth waits the longest that a script with that many workers
        asks: of 20 idle workers, where one script has 10 wait 100 s and
        another has 20 wait 50 s, 10 leave after 50 s. With no script
        connected, the last to ask counts. A local pool's threads always stay.
        """
        pool = allocation.pool
        mine = [a for a in self.allocations if a.pool == pool]
        idle = sorted((a for a in mine if not a.used), key=lambda a: a.idle_since)
        rank = len(mine) - idle.index(allocation)
        waits = [
            math.inf if isinstance(asked, LocalPool) else asked.idle_timeout_seconds
            for asked in self.asked(pool) or [self.last_asked[pool]]
            if isinstance(asked, LocalPool) or asked.max_allocations >= rank
        ]
        return max(waits, default=0.0) + allocation.idle_since - time.monotonic()

    def _job(self, specs: list[str], pool: str, *, lease: int, slots: int) -> Job:
        """The job that runs ``specs``: their lines and the lines their fields reach."""
        placed = self.placed[specs[0]]
        asked = self.asked_by[specs[0]][0].pools.get(pool)
        lines = placed.plan.lines
        reached = dict.fromkeys(specs)
        pending = list(specs)
        while pending:
            for key in plan.holds(lines[pending.pop()], returned=False):
                # What a serializer wrote can look like a pointer and have no line.
                if key not in reached and key in lines:
                    reached[key] = None
                    pending.append(key)
        if "moved" in lines:
            reached["moved"] = None
        return Job(
            lease=lease,
            slots=slots,
            specs=specs,
            lines=[lines[key] for key in reached],
            class_dir=placed.class_dir,
            process=placed.process,
            slurm_job_name=asked.job_name if isinstance(asked, SlurmPool) else None,
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
        first spec. A batch whose working folder was not free to take when it
        was last looked at waits: its worker is alive, or nothing says that it
        is dead.
        """
        for stopped in list(self.stopped):
            if (
                stopped.why is None
                and not stopped.held
                and not any(map(self.dag.waits, stopped.specs))
                and wanted(stopped.specs[0])
            ):
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

        script = self.asked_by[specs[0]][0]
        label = _group_label(specs[0], len(specs))
        run = run_name([spec.rpartition(":")[2] for spec in specs])
        for stopped in used:
            if stopped.run != run:
                reason = (
                    f"{label} starts in a new working folder: it holds specs of a "
                    "batch that stopped and is not that batch alone, so what the "
                    "batch had done is dropped. Its working folder stays in "
                    f"attempts/: {attempt_parent_in(placed.class_dir, stopped.run)}"
                )
                logger.warning("%s", reason)
                self.outbox.append((script, WarningMessage(reason=reason)))
        if len(specs) < batch.cap:
            for other in self.unstarted.get(pool, {}).get(batch, {}):
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

    def asked(self, pool: str) -> list[Pool]:
        """What each connected script asks of ``pool``'s workers, newest last."""
        with self.lock:
            return [s.pools[pool] for s in self.scripts if pool in s.pools]

    def demand(self, pool: str) -> int:
        """Workers ``pool`` should have: the busy ones, and the fewest more that
        hold the jobs it could start, packed first-fit decreasing by slots."""
        with self.lock:
            asked = self.asked(pool)
            most = max(map(_most, asked), default=0)
            slots = asked[-1].slots if asked and isinstance(asked[-1], SlurmPool) else 1
            loads = [a.used for a in self.allocations if a.pool == pool and a.used]
            # Past this many slots, more jobs change nothing.
            room = most * slots - sum(loads)
            running = Counter(
                lease.throttle for lease in self.leases.values() if lease.throttle
            )
            jobs = [
                self.placed[stopped.specs[0]].pools[pool].slots
                for stopped in self.stopped
                if not stopped.held
                and pool in self.placed[stopped.specs[0]].pools
                and not any(map(self.dag.waits, stopped.specs))
            ]
            room -= sum(jobs)
            for plans in self.unstarted.get(pool, {}).values():
                for queue in plans.values():
                    if room <= 0:
                        break
                    placed = self.placed[next(iter(queue))]
                    batch = placed.pools[pool]
                    wanted = min(-(-len(queue) // batch.cap), -(-room // batch.slots))
                    if (throttle := placed.throttle) is not None:
                        free = throttle.max_running - running[throttle.key]
                        wanted = max(0, min(wanted, free))
                        running[throttle.key] += wanted
                    jobs += [batch.slots] * wanted
                    room -= batch.slots * wanted
            for job in sorted(jobs, reverse=True):
                for i, load in enumerate(loads):
                    if load + job <= slots:
                        loads[i] += job
                        break
                else:
                    if len(loads) >= most:
                        break
                    loads.append(job)
            return min(len(loads), most)


def _most(pool: Pool) -> int:
    """How many workers a script asks ``pool`` to have at most."""
    return pool.max_workers if isinstance(pool, LocalPool) else pool.max_allocations


def _peer(connection: ServerConnection) -> str:
    with suppress(OSError, TypeError):
        host, port, *_ = connection.remote_address
        return f"{host}:{port}"
    return "?"
