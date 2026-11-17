from __future__ import annotations

import errno
import shutil
import time
import traceback
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, overload

from furu._batched import _BatchedHook
from furu._declared_types import declared_result_type
from furu._tree import map_specs, specs_in
from furu.code_trace import (
    LogState,
    Recording,
    Trace,
    build_trace,
    recording,
    why_not_resumable,
)
from furu.core import Missing, Spec
from furu.dag import Dag
from furu.dependencies import (
    _DependencyNotReady,
    any_under_creation,
    missing_dependencies,
    record_loaded,
    recording_loads,
    under_creation,
)
from furu.execution.launch import ask, asked, held_in
from furu.locking import lock
from furu.logging import _group_label, _log_scope, get_logger
from furu.migration.schema import check_defaults, record_nested_names
from furu.provenance import (
    ExecuteContext,
    Provenance,
    SubmitProvenance,
    _require_uv,
    capture_submit_provenance,
)
from furu.resources import Worker
from furu.result.bundle import _save_result_bundle
from furu.storage import events
from furu.storage._layout import (
    FIXED,
    attempts_dir_in,
    data_dir_in,
    events_path_in,
    lock_path_in,
    log_path_in,
    provenance_path_in,
    result_dir_in,
    run_dir_in,
    scratch_dir_in,
    trace_log_path_in,
    trace_path_in,
)
from furu.storage.stored import load_stored_result, stored_run, why_rerun
from furu.utils import atomic_write_text, format_duration

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from pydantic import BaseModel

    from furu.worker.backends.protocol import WorkerBackend

type HasLock = Callable[[], bool]


def _open_attempt(obj: Spec) -> LogState:
    """Start an attempt in obj's working folder; return what earlier ones ran.

    A fixed spec continues with what an earlier attempt left there. A traced
    one does only if the code that attempt ran (recorded in trace.log) is
    unchanged on disk; otherwise the files it wrote are dropped.
    """
    metadata = obj._metadata
    attempt = obj._attempt_dir
    if why := why_rerun(obj):
        obj.logger.info("rerunning %s: %s", obj._log_label, why)
    resumed = LogState()
    if metadata.code_version == "traced" and events_path_in(attempt).exists():
        logged = LogState.read(trace_log_path_in(attempt))
        if why := why_not_resumable(logged, storage_root=metadata.storage):
            obj.logger.info("starting %s over: %s", obj._log_label, why)
            shutil.rmtree(data_dir_in(attempt), ignore_errors=True)
            shutil.rmtree(scratch_dir_in(attempt), ignore_errors=True)
            trace_log_path_in(attempt).unlink(missing_ok=True)
        else:
            assert logged is not None
            resumed = logged
    events.start(attempt, [obj._artifact_hash])
    return resumed


@contextmanager
def _recording(objs: Sequence[Spec], resumed: LogState) -> Generator[Recording]:
    """Trace one create() call and note the code versions of what it loads."""
    metadata = objs[0]._metadata  # a batch shares its metadata
    with (
        recording(
            [trace_log_path_in(obj._attempt_dir) for obj in objs],
            code_version=metadata.code_version,
            storage_root=metadata.storage,
            resumed=resumed,
        ) as run,
        recording_loads(run.loaded),
        under_creation(objs),
    ):
        yield run


def _call_create[T](
    group: list[Spec[T]], resumed: list[LogState]
) -> tuple[list[T], list[Recording]]:
    """Run the group's create(): one call for a batch, else one per spec."""
    cls = type(group[0])
    match getattr(cls, "_furu_create_hook", None):
        case None:
            raise TypeError(
                f"{cls.__qualname__} cannot create missing results "
                "because it does not define create()"
            )
        case _BatchedHook(func=create_hook):
            union = LogState()
            for state in resumed:
                union.merge(state)
            with _recording(group, union) as run:
                results = create_hook(group)
            if not isinstance(results, list):
                raise TypeError(f"{cls.__name__}.create() must return a list")
            if len(results) != len(group):
                raise TypeError(
                    f"{cls.__name__} returned {len(results)} results for "
                    f"{len(group)} objects"
                )
            # One create() call built the whole batch, so every object
            # records the code it ran and the versions it loaded.
            return results, [run] * len(group)
        case create_hook:
            results = []
            runs = []
            for obj, state in zip(group, resumed, strict=True):
                with _recording([obj], state) as run:
                    results.append(create_hook(obj))
                runs.append(run)
            return results, runs


def _store[T](
    obj: Spec[T],
    result: T,
    *,
    run: Recording,
    has_lock: HasLock,
    submit_provenance: SubmitProvenance,
    started_at: datetime,
) -> Trace:
    """Write the result into obj's working folder; return the code that made it."""
    attempt = obj._attempt_dir
    if not has_lock():
        raise RuntimeError(f"lost the lock on {attempt} before writing the result")

    result_dir = result_dir_in(attempt)
    shutil.rmtree(result_dir, ignore_errors=True)  # left by a crash before publishing
    _save_result_bundle(
        [result],
        result_dir,
        declared_type=declared_result_type(type(obj)),
        result_codecs=obj.result_codecs,
        data_dir=data_dir_in(attempt),
        label=type(obj).__name__,
    )
    trace = build_trace(run, label=obj._log_label, warn=obj.logger.warning)
    provenance = Provenance.merge(
        submit_provenance, ExecuteContext.capture(started_at=started_at)
    )
    atomic_write_text(provenance_path_in(attempt), provenance.model_dump_json(indent=2))
    shutil.rmtree(scratch_dir_in(attempt), ignore_errors=True)
    trace_log_path_in(attempt).unlink(missing_ok=True)
    return trace


def _publish(obj: Spec, trace: Trace, *, has_lock: HasLock) -> None:
    """Rename obj's working folder into runs/ of its code version, name unchanged.

    The lock is held through the rename, so nobody else is in the folder, and
    readers see the whole run or none of it. A run that is already there is
    never replaced.
    """
    attempt = obj._attempt_dir
    version = obj._class_dir / trace.version
    run = run_dir_in(version, obj._run_name)
    run.parent.mkdir(parents=True, exist_ok=True)
    if trace.version != FIXED and not (trace_path := trace_path_in(version)).exists():
        atomic_write_text(
            trace_path, trace.model_dump_json(indent=2, exclude_none=True)
        )
    if not has_lock():
        raise RuntimeError(f"lost the lock on {attempt} before publishing")
    obj.logger.debug("publishing %s under %s", obj._log_label, version.name)
    try:
        attempt.rename(run)
    except OSError as exc:
        if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
            raise
        raise RuntimeError(
            f"{obj._log_label} is already published at {run}; its working "
            f"folder {attempt} was left as it is"
        ) from exc


@overload
def _load_or_create[T](
    tree: Spec[T], *, on: Sequence[WorkerBackend] | None = None, load: bool = True
) -> T: ...
@overload
def _load_or_create(
    tree: object, *, on: Sequence[WorkerBackend] | None = None, load: bool = True
) -> Any: ...
def _load_or_create(
    tree: object, *, on: Sequence[WorkerBackend] | None = None, load: bool = True
) -> Any:
    if on is not None:
        from furu.execution.run_on_workers import run_on_workers

        run_on_workers(specs_in(tree), on)
    _require_uv()
    if isinstance(tree, Spec):
        tree.logger.debug(".create called for %s", tree)
    objs = specs_in(tree)
    if any_under_creation():
        outputs = _load_or_block(objs, load=load)
    else:
        outputs = _load_or_create_local(objs, load=load)
    if not load:
        return None
    results = {obj.object_id: output for obj, output in zip(objs, outputs)}
    return map_specs(lambda obj: results[obj.object_id], tree)


def _ensure_group_result[T](
    objs: Sequence[Spec[T]], *, submit_provenance: SubmitProvenance
) -> None:
    missing: list[Spec[T]] = []
    for obj in objs:
        if stored_run(obj) is not None:
            obj.logger.info("cache hit for %s", obj._log_label)
            continue
        class_dir = obj._class_dir
        attempts_dir_in(class_dir).mkdir(parents=True, exist_ok=True)
        check_defaults(class_dir, type(obj), record=True)
        record_nested_names(class_dir, held_in(obj))
        missing.append(obj)

    if not missing:
        return

    with lock(
        [lock_path_in(obj._class_dir, obj._run_name) for obj in missing]
    ) as has_lock:
        # Whoever held the lock before may have published the run and taken
        # its working folder with it.
        pending = [obj for obj in missing if stored_run(obj) is None]
        if pending:
            _create_and_store_group(
                pending,
                has_lock=has_lock,
                submit_provenance=submit_provenance,
            )


# Python cannot map Spec[T] -> T through an arbitrary structure, so the shapes
# it can express get exact types and every other pytree is Any. Dataclasses and
# pydantic models come back as dicts of their fields.
@overload
def create[T](tree: Spec[T], /, *, on: Sequence[WorkerBackend] | None = None) -> T: ...
@overload
def create[T0, T1](
    tree: tuple[Spec[T0], Spec[T1]], /, *, on: Sequence[WorkerBackend] | None = None
) -> tuple[T0, T1]: ...
@overload
def create[T0, T1, T2](
    tree: tuple[Spec[T0], Spec[T1], Spec[T2]],
    /,
    *,
    on: Sequence[WorkerBackend] | None = None,
) -> tuple[T0, T1, T2]: ...
@overload
def create[T](
    tree: tuple[Spec[T], ...], /, *, on: Sequence[WorkerBackend] | None = None
) -> tuple[T, ...]: ...
@overload
def create[T](
    tree: Sequence[Spec[T]], /, *, on: Sequence[WorkerBackend] | None = None
) -> list[T]: ...
@overload
def create[K, T](
    tree: Mapping[K, Spec[T]], /, *, on: Sequence[WorkerBackend] | None = None
) -> dict[K, T]: ...
@overload
def create(
    tree: DataclassInstance | BaseModel,
    /,
    *,
    on: Sequence[WorkerBackend] | None = None,
) -> dict[str, Any]: ...
@overload
def create(tree: object, /, *, on: Sequence[WorkerBackend] | None = None) -> Any: ...
def create(tree: object, /, *, on: Sequence[WorkerBackend] | None = None) -> Any:
    return _load_or_create(tree, on=on)


def build(tree: object, /, *, on: Sequence[WorkerBackend] | None = None) -> None:
    """Like create(), but leaves the results on disk instead of loading them."""
    _load_or_create(tree, on=on, load=False)


@overload
def load_existing[T](tree: Spec[T], /) -> T: ...
@overload
def load_existing[T0, T1](tree: tuple[Spec[T0], Spec[T1]], /) -> tuple[T0, T1]: ...
@overload
def load_existing[T0, T1, T2](
    tree: tuple[Spec[T0], Spec[T1], Spec[T2]], /
) -> tuple[T0, T1, T2]: ...
@overload
def load_existing[T](tree: tuple[Spec[T], ...], /) -> tuple[T, ...]: ...
@overload
def load_existing[T](tree: Sequence[Spec[T]], /) -> list[T]: ...
@overload
def load_existing[K, T](tree: Mapping[K, Spec[T]], /) -> dict[K, T]: ...
@overload
def load_existing(tree: DataclassInstance | BaseModel, /) -> dict[str, Any]: ...
@overload
def load_existing(tree: object, /) -> Any: ...
def load_existing(tree: object, /) -> Any:
    objs = specs_in(tree)
    loaded: dict[str, Any] = {}
    missing: list[Spec] = []
    for obj in objs:
        if (run := stored_run(obj)) is None:
            missing.append(obj)
            continue
        record_loaded(run)
        loaded[obj.object_id] = load_stored_result(obj, run)
    if missing:
        first = missing[0]
        raise Missing(
            f"{first._log_label}.load_existing() could not find a result. "
            "load_existing() only loads existing results; use create() to compute "
            "missing results."
        )
    if objs:
        get_logger().info(
            "loaded %d furu objects including %s", len(objs), objs[0]._log_label
        )
    else:
        get_logger().info("loaded 0 furu objects")
    return map_specs(lambda obj: loaded[obj.object_id], tree)


def _cached_to_build_msg(cached: list[Spec], to_build: list[Spec]) -> str:
    def fmt(objs: list[Spec]) -> str:
        if len(cached) + len(to_build) > 5:
            return str(len(objs))
        return ", ".join(o._log_label for o in objs)

    msg = f"cached {fmt(cached)}"
    return f"building {fmt(to_build)}, {msg}" if to_build else msg


def _load_or_block[T](objs: list[Spec[T]], *, load: bool) -> list[T]:
    """Load ``objs`` inside a create hook, or block that create if any are missing.

    Blocking aborts the hook; the caller builds the missing specs and reruns it
    from scratch, so specs never build inside another spec's create.
    """
    loaded: list[T] = []
    cached: list[Spec[T]] = []
    missing: list[Spec[T]] = []

    for obj in objs:
        if (run := stored_run(obj)) is not None:
            record_loaded(run)
            if load:
                loaded.append(load_stored_result(obj, run))
            cached.append(obj)
        else:
            missing.append(obj)

    if cached:
        objs[0].logger.info("%s", _cached_to_build_msg(cached, missing))

    if missing:
        # What a launch would list for them: the classes under them can have
        # changed too.
        dag = Dag()
        dag.add(missing)
        ask([node.obj for node in dag.nodes_by_id.values()], found=True)
        raise _DependencyNotReady(missing)

    return loaded


def _load_or_create_local[T](objs: list[Spec[T]], *, load: bool) -> list[T]:
    """Build what ``objs`` need in this process, then return their results.

    Runs the same graph as a worker run: a create that blocks on missing specs
    is aborted with its locks released, they are built, and it reruns.
    """
    dag = Dag()
    dag.add(objs)
    cached = [obj for obj in objs if obj.object_id not in dag.nodes_by_id]
    if cached:
        to_build = [obj for obj in objs if obj.object_id in dag.nodes_by_id]
        objs[0].logger.info("%s", _cached_to_build_msg(cached, to_build))

    if dag.ready:
        with asked(ask([node.obj for node in dag.nodes_by_id.values()])):
            _run_in_process(dag)
    if not load:
        return []
    # Load the way a cache hit does, so the first run returns what every later
    # load will. Specs sharing an object_id are one spec whatever their storage;
    # like the dag, the first one stands in for the rest.
    first: dict[str, Spec[T]] = {}
    for obj in objs:
        first.setdefault(obj.object_id, obj)
    loaded = {object_id: obj.load_existing() for object_id, obj in first.items()}
    return [loaded[obj.object_id] for obj in objs]


def _run_in_process(dag: Dag) -> None:
    """Create every ready spec in ``dag``."""
    submit_provenance = capture_submit_provenance()
    here = Worker.here()
    while dag.ready:
        lead = next(iter(dag.ready.values()))
        group = [lead]
        if (batch := lead.batch_group(here)) is not None:
            group = [n for n in dag.ready.values() if n.batch_group(here) == batch]
            group = group[: batch[1]]
        for node in group:
            del dag.ready[node.obj.object_id]

        try:
            _create_in_process(
                [node.obj for node in group], submit_provenance=submit_provenance
            )
        except _DependencyNotReady as exc:
            lead.obj.logger.info(
                "blocked %s · %s",
                _group_label(lead.obj._log_label, len(group)),
                missing_dependencies(len(exc.dependencies)),
            )
            for node in group:
                if (why := dag.block(node, exc.dependencies)) is not None:
                    raise RuntimeError(f"{node.obj._log_label} {why}") from None
            continue
        except Exception as exc:
            if lead.dependents:
                exc.add_note(
                    "while building dependencies of "
                    + lead.dependents[0].obj._log_label
                )
            raise
        for node in group:
            dag.complete(node)


def _create_in_process[T](
    group: list[Spec[T]], *, submit_provenance: SubmitProvenance
) -> None:
    """Create one group in this process, announced on the terminal."""
    lead = group[0]
    label = _group_label(lead._log_label, len(group))
    lead.logger.info(
        "creating %s", label, extra={"path": log_path_in(lead._attempt_dir)}
    )
    started_at = time.monotonic()
    _ensure_group_result(group, submit_provenance=submit_provenance)
    duration = format_duration(time.monotonic() - started_at)
    lead.logger.info("finished %s ok · %s", label, duration)


def _create_and_store_group[T](
    group: list[Spec[T]],
    *,
    has_lock: HasLock,
    submit_provenance: SubmitProvenance,
) -> None:
    """Run create() for ``group`` in its working folders, then publish them.

    Each folder's events.jsonl gets this attempt's start and end. What
    create() writes goes to the log in the lead's folder: one call, one log.
    """
    started_at = datetime.now(UTC)
    lead = group[0]
    attempts = [obj._attempt_dir for obj in group]
    for attempt in attempts:
        attempt.mkdir(exist_ok=True)
    with _log_scope(log_path_in(attempts[0])):
        resumed = [_open_attempt(obj) for obj in group]
        try:
            results, runs = _call_create(group, resumed)
            traces = [
                _store(
                    obj,
                    result,
                    run=run,
                    has_lock=has_lock,
                    submit_provenance=submit_provenance,
                    started_at=started_at,
                )
                for obj, result, run in zip(group, results, runs, strict=True)
            ]
        except _DependencyNotReady as exc:
            waiting_on = [dep.object_id for dep in exc.dependencies]
            for attempt in attempts:
                events.end(attempt, "blocked", waiting_on=waiting_on)
            raise
        except BaseException as exc:
            if isinstance(exc, Exception):
                lead.logger.exception("create failed for %s", lead._log_label)
            error = "".join(traceback.format_exception_only(exc)).splitlines()[0]
            for attempt in attempts:
                events.end(attempt, "failed", error=error)
            raise
        for obj, trace in zip(group, traces, strict=True):
            events.end(obj._attempt_dir, "ok")
            _publish(obj, trace, has_lock=has_lock)
