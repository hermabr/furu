"""The furu manager: one queue and one set of worker pools for every launch.

It holds only memory. A script sends its plan's lines and its pool
definitions on every connection, so a restarted manager needs no state file:
scripts plan again from disk and resend, and the working folders in
``attempts/`` say which batches were started and who holds each. Run it with
``uvx furu@<version> manager``.

It forms a batch when a worker asks: specs nobody has started, then stopped
batches that fit whole, then as much of the oldest stopped batch as there is
room for. A stopped batch is one that was handed out and did not finish. Its
working folder is named for its specs in order, so it is continued only when
it goes out alone.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from itertools import islice
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
from furu.storage._layout import attempt_parent_in, attempts_dir_in, run_name
from furu.storage.events import read_events
from furu.worker.backends.local import LocalThreadPool
from furu.worker.backends.protocol import WorkerPool
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.messages import (
    Batch,
    CancelMessage,
    ErrorMessage,
    GaveUpMessage,
    HelloMessage,
    Job,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
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
    job_result_adapter,
    script_message_adapter,
)

logger = get_logger("manager")

# How often an idle worker's handler checks that the worker is still there.
_IDLE_CHECK_INTERVAL_S = 5.0
# How often the working folder of a stopped batch is looked at again.
_HELD_CHECK_INTERVAL_S = 30.0
# Not part of a pool's key: its first script sets them and later ones must agree.
_SETTINGS = {"job_name", "idle_timeout_seconds"}

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
    # Whether its working folder was free to take when last looked at, when it
    # is looked at next, and what its script was told if it was another's.
    free: bool = False
    look_at: float = 0.0
    why: str | None = None


@dataclass(eq=False, slots=True)
class _Lease:
    specs: list[str]
    job: Job
    pool: str
    throttle: str | None
    started: float


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
    gave_up: set[str] = field(default_factory=set)
    leases: dict[ServerConnection, _Lease] = field(default_factory=dict)
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
            clashes = [
                f"pool {key} already runs with {first}, not {asked}"
                for key, definition in message.pools.items()
                if key in self.pools
                and (first := self.pools[key].definition.model_dump(include=_SETTINGS))
                != (asked := definition.model_dump(include=_SETTINGS))
            ]
            if not clashes:
                script.pools = message.pools
                for key, definition in message.pools.items():
                    if key not in self.pools:
                        self.pools[key] = self._start_pool(key, definition)
                queued = self._add(script, message)
                self.lock.notify_all()
        if clashes:
            hint = "; restart the manager to change them"
            script.send(ErrorMessage(reason="; ".join(clashes) + hint))
        elif queued:
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
                worker
                for worker, lease in self.leases.items()
                if any(script in self.asked_by.get(spec, ()) for spec in lease.specs)
            ]
            dropped = self._drop(script)
            workers = [
                worker
                for worker in running
                if not any(spec in self.placed for spec in self.leases[worker].specs)
            ]
            self.lock.notify_all()
        logger.info(
            "cancelled %d queued and %d running jobs from %s",
            dropped,
            len(workers),
            _peer(script.connection),
        )
        for worker in workers:
            with suppress(ConnectionClosed):
                worker.send(CancelMessage().model_dump_json())

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
        worker = hello.worker
        with self.lock:
            known = hello.pool in self.pools
        if not known:
            logger.warning("worker %s belongs to no pool here; closing", worker)
            return
        logger.debug("worker connected · %s", worker)
        while (lease := self._lease(connection, pool=hello.pool)) is not None:
            job = lease.job
            logger.info(
                "leased %s to %s", job.label, worker, extra={"path": job.folder}
            )
            result: JobResult
            try:
                connection.send(job.model_dump_json())
                result = job_result_adapter.validate_json(connection.recv())
            except ConnectionClosed:  # the next _lease sees it and returns None
                if self.stopping:  # the worker finishes its job alone; it hasn't failed
                    return
                logger.warning("lost %s mid-job", worker, extra={"path": hello.log})
                result = JobFailedResult(error=f"worker {worker} disconnected")
            with self.lock:
                del self.leases[connection]
                if isinstance(result, JobCompletedResult):
                    self.pools[lease.pool].job_completed()
                self._finish(lease, result)
                self.lock.notify_all()
            self._flush()

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

    def _lease(self, connection: ServerConnection, *, pool: str) -> _Lease | None:
        while True:
            with self.lock:
                if self.stopping or connection.state is not State.OPEN:
                    return None
                # Looked at first: a job is filled from the stopped batches
                # whose working folders are free.
                due = self._due(pool)
            if due:
                self._look(due)
                self._flush()  # a look can leave a warning
            with self.lock:
                if self.stopping or connection.state is not State.OPEN:
                    return None
                lease = self._hand_out(connection, pool)
                if lease is None:
                    self.lock.wait(timeout=_IDLE_CHECK_INTERVAL_S)
            self._flush()  # forming a batch can leave a warning
            if lease is not None:
                return lease

    def _due(self, pool: str) -> list[tuple[_Stopped, Path]]:
        """The stopped batches a worker of ``pool`` waits for whose working
        folders are looked at now, each with the folder that holds its."""
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
                due.append((stopped, attempt_parent_in(placed.class_dir, stopped.run)))
        return due

    def _look(self, due: list[tuple[_Stopped, Path]]) -> None:
        """Find out whether the working folder of each batch in ``due`` is free.

        Asked without the lock: asking about a Slurm job's folder runs Slurm
        commands. The script that answers for a batch whose folder is
        another's is told why, once for each reason.
        """
        whys = [attempt_lock.held(folder) for _, folder in due]
        next_look = time.monotonic() + _HELD_CHECK_INTERVAL_S
        with self.lock:
            for (stopped, _), why in zip(due, whys):
                stopped.look_at = next_look
                stopped.free = why is None
                # Not if its specs finished or were given up on meanwhile.
                if why not in (None, stopped.why) and stopped in self.stopped:
                    label = _group_label(stopped.specs[0], len(stopped.specs))
                    reason = f"{label} waits: {why}"
                    logger.warning("%s", reason)
                    script = self.asked_by[stopped.specs[0]][0]
                    self.outbox.append((script, WarningMessage(reason=reason)))
                stopped.why = why
            self.lock.notify_all()

    def _hand_out(self, connection: ServerConnection, pool: str) -> _Lease | None:
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
        throttle = self.placed[specs[0]].throttle
        lease = self.leases[connection] = _Lease(
            specs,
            self._job(specs),
            pool,
            None if throttle is None else throttle.key,
            time.monotonic(),
        )
        return lease

    def _job(self, specs: list[str]) -> Job:
        """The job that runs ``specs``: their lines and the lines their fields reach."""
        placed = self.placed[specs[0]]
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
        first spec. A batch whose working folder was not free to take when it
        was last looked at waits: its worker is alive, or nothing says that it
        is dead.
        """
        for stopped in list(self.stopped):
            if (
                stopped.free
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
                    f"attempts/: {attempt_parent_in(placed.class_dir, stopped.run)}"
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

    def demand(self, pool: str) -> int:
        """Workers ``pool`` should have: busy ones plus jobs it could start."""
        with self.lock:
            max_workers = max(
                (
                    script.pools[pool].max_workers
                    for script in self.scripts
                    if pool in script.pools
                ),
                default=0,
            )
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


def _peer(connection: ServerConnection) -> str:
    with suppress(OSError, TypeError):
        host, port, *_ = connection.remote_address
        return f"{host}:{port}"
    return "?"
