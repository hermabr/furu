from __future__ import annotations

import secrets
import shutil
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

from furu import plan, record
from furu._batched import _BatchedHook, batch_of
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
from furu.config import get_config
from furu.core import Missing, Spec, working_folder
from furu.dag import Dag, why_stuck
from furu.dependencies import (
    _DependencyNotReady,
    any_under_creation,
    missing_dependencies,
    record_loaded,
    recording_loads,
    under_creation,
)
from furu.execution.launch import ask, asked, held_in
from furu.execution.launch import missing as missing_under
from furu.locking import lock
from furu.logging import _group_label, _log_scope, get_logger
from furu.migration.schema import check_defaults, record_nested_names
from furu.provenance import _require_uv, capture_submit_provenance
from furu.resources import Worker
from furu.result.bundle import _save_result_bundle
from furu.serializer.artifact import remembering
from furu.storage import events
from furu.storage._layout import (
    FIXED,
    attempts_dir_in,
    data_dir_in,
    events_path_in,
    lock_path_in,
    log_path_in,
    result_dir_in,
    run_dir_in,
    scratch_dir_in,
    trace_log_path_in,
    trace_path_in,
)
from furu.storage.directory import SpecDirectory
from furu.storage.stored import Row, Stored, load_rows, why_rerun
from furu.utils import atomic_write_text, format_duration

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from pydantic import BaseModel

    from furu.worker.backends.protocol import WorkerBackend

type HasLock = Callable[[], bool]


def _open_attempt(group: Sequence[Spec], attempt: Path) -> LogState:
    """What earlier attempts in the group's working folder ran, for a new one to go on from.

    A fixed class continues with what an earlier attempt left there. A traced
    one does only if the code that attempt ran (recorded in trace.log) is
    unchanged on disk; otherwise the files it wrote are dropped.
    """
    lead = group[0]
    metadata = lead._metadata
    label = _group_label(lead._log_label, len(group))
    traced = metadata.code_version == "traced"
    if traced and (why := why_rerun(lead._class_dir, attempt.name)):
        lead.logger.info("rerunning %s: %s", label, why)
    resumed = LogState()
    if traced and events_path_in(attempt).exists():
        logged = LogState.read(trace_log_path_in(attempt))
        if why := why_not_resumable(logged, storage_root=metadata.storage):
            lead.logger.info("starting %s over: %s", label, why)
            shutil.rmtree(data_dir_in(attempt), ignore_errors=True)
            shutil.rmtree(scratch_dir_in(attempt), ignore_errors=True)
            trace_log_path_in(attempt).unlink(missing_ok=True)
        else:
            assert logged is not None
            resumed = logged
    return resumed


@contextmanager
def _recording(
    group: Sequence[Spec], attempt: Path, resumed: LogState
) -> Generator[tuple[Recording | None, list[Spec]]]:
    """Note the specs whose results one create() call loads, and trace the call.

    Only a class that traces its code is traced, with the code versions of
    what it loads.
    """
    metadata = group[0]._metadata  # a batch has one Metadata
    loaded: list[Spec] = []
    with (
        nullcontext()
        if metadata.code_version == "fixed"
        else recording(
            trace_log_path_in(attempt), storage_root=metadata.storage, resumed=resumed
        )
    ) as run:

        def on_load(folder: Path, specs: Iterable[Spec]) -> None:
            loaded.extend(specs)
            if run is not None:
                run.loaded(folder)

        with recording_loads(on_load), under_creation(group):
            yield run, loaded


def _call_create[T](
    group: list[Spec[T]], attempt: Path, resumed: LogState
) -> tuple[list[T], Recording | None, list[Spec]]:
    """Run the group's one create() call: on the whole batch, or on its one spec.

    Returns its results, its trace recording when its class traces its code,
    and the specs whose results it loaded.
    """
    cls = type(group[0])
    hook = getattr(cls, "_furu_create_hook", None)
    if hook is None:
        raise TypeError(
            f"{cls.__qualname__} cannot create missing results "
            "because it does not define create()"
        )
    with _recording(group, attempt, resumed) as (run, loaded):
        if isinstance(hook, _BatchedHook):
            results = hook.func(group, SpecDirectory(attempt))
        else:
            (obj,) = group
            results = [hook(obj)]
    if not isinstance(results, list):
        raise TypeError(f"{cls.__name__}.create() must return a list")
    if len(results) != len(group):
        raise TypeError(
            f"{cls.__name__} returned {len(results)} results for {len(group)} objects"
        )
    return results, run, loaded


def _store[T](
    group: list[Spec[T]],
    results: list[T],
    attempt: Path,
    *,
    run: Recording | None,
    has_lock: HasLock,
) -> Trace | None:
    """Write the results into the working folder, a row per spec; return the trace, if traced."""
    if not has_lock():
        raise RuntimeError(f"lost the lock on {attempt} before writing the result")

    lead = group[0]
    result_dir = result_dir_in(attempt)
    shutil.rmtree(result_dir, ignore_errors=True)  # left by a crash before publishing
    _save_result_bundle(
        results,
        result_dir,
        declared_type=declared_result_type(type(lead)),
        result_codecs=lead.result_codecs,
        data_dir=data_dir_in(attempt),
        label=type(lead).__name__,
    )
    trace = None
    if run is not None:
        trace = build_trace(
            run,
            label=_group_label(lead._log_label, len(group)),
            warn=lead.logger.warning,
        )
    shutil.rmtree(scratch_dir_in(attempt), ignore_errors=True)
    trace_log_path_in(attempt).unlink(missing_ok=True)
    return trace


def _publish(group: Sequence[Spec], trace: Trace | None, *, has_lock: HasLock) -> None:
    """Rename the group's working folder into runs/ of its code version, name unchanged.

    The lock is held through the rename, so nobody else is in the folder, and
    readers see the whole run or none of it. A run that is already there is
    never replaced.
    """
    lead = group[0]
    attempt = working_folder(group)
    version = lead._class_dir / (FIXED if trace is None else trace.version)
    run = run_dir_in(version, attempt.name)
    run.parent.mkdir(parents=True, exist_ok=True)
    if trace is not None:
        trace_path = trace_path_in(version)
        try:
            Trace.model_validate_json(trace_path.read_bytes())
        except (OSError, ValueError):  # missing, or cut short when it was written
            atomic_write_text(
                trace_path, trace.model_dump_json(indent=2, exclude_none=True)
            )
    if not has_lock():
        raise RuntimeError(f"lost the lock on {attempt} before publishing")
    lead.logger.debug(
        "publishing %s under %s",
        _group_label(lead._log_label, len(group)),
        version.name,
    )
    attempt.rename(run)


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
    # One launch: a large value that many of its specs hold is named once.
    with remembering():
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


def _ensure_group_result[T](objs: Sequence[Spec[T]]) -> None:
    """Create and publish one create() call's ``objs`` as one run, unless all are stored.

    The run and its working folder are named for all of ``objs``, so one that
    is stored already is computed again with the rest.
    """
    look = Stored()
    if all(look.row(obj) is not None for obj in objs):
        objs[0].logger.info(
            "cache hit for %s", _group_label(objs[0]._log_label, len(objs))
        )
        return

    class_dir = objs[0]._class_dir
    attempts_dir_in(class_dir).mkdir(parents=True, exist_ok=True)
    check_defaults(class_dir, type(objs[0]), record=True)
    for obj in objs:
        record_nested_names(class_dir, held_in(obj))

    with lock(lock_path_in(class_dir, working_folder(objs).name)) as has_lock:
        # Whoever held the lock before may have published the run and taken
        # its working folder with it.
        look = Stored()
        if any(look.row(obj) is None for obj in objs):
            _create_and_store_group(list(objs), has_lock=has_lock)


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
    loaded = _load(objs)
    if objs:
        get_logger().info(
            "loaded %d furu objects including %s", len(objs), objs[0]._log_label
        )
    else:
        get_logger().info("loaded 0 furu objects")
    return map_specs(lambda obj: loaded[obj.object_id], tree)


def _load(objs: Sequence[Spec]) -> dict[str, Any]:
    """The stored result of each spec, by object id, reading every run once.

    Specs sharing an object_id are one spec whatever their storage; like the
    dag, the first one stands in for the rest.
    """
    first: dict[str, Spec] = {}
    for obj in objs:
        first.setdefault(obj.object_id, obj)
    look = Stored()
    rows = []
    for obj in first.values():
        if (row := look.row(obj)) is None:
            raise Missing(
                f"{obj._log_label}.load_existing() could not find a result. "
                "load_existing() only loads existing results; use create() to "
                "compute missing results."
            )
        rows.append(row)
    return dict(zip(first, load_rows(list(first.values()), rows), strict=True))


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
    look = Stored()
    rows = [look.row(obj) for obj in objs]
    cached = [obj for obj, row in zip(objs, rows) if row is not None]
    missing = [obj for obj, row in zip(objs, rows) if row is None]

    if cached:
        objs[0].logger.info("%s", _cached_to_build_msg(cached, missing))

    if missing:
        # What a launch would list for them: the classes under them can have
        # changed too.
        running = missing_under(missing, Dag())
        ask(running, found=True)
        # Saved before anyone is told, so a found spec that fails or never
        # starts is still described.
        raise _DependencyNotReady(missing, *record.found(running))

    found = cast(list[Row], rows)
    if load:
        return load_rows(objs, found)
    for obj, row in zip(objs, found):
        record_loaded(row.run, [obj])
    return []


def _load_or_create_local[T](objs: list[Spec[T]], *, load: bool) -> list[T]:
    """Build what ``objs`` need in this process, then return their results.

    Runs the same graph as a worker run: a create that blocks on missing specs
    is aborted with its locks released, they are built, and it reruns.
    """
    dag = Dag()
    running = missing_under(objs, dag)
    cached = [obj for obj in objs if obj.object_id not in dag.nodes]
    if cached:
        to_build = [obj for obj in objs if obj.object_id in dag.nodes]
        objs[0].logger.info("%s", _cached_to_build_msg(cached, to_build))

    if running:
        launch_plan = (
            get_config().directories.anchored().plans / f"{secrets.token_hex(16)}.plan"
        )
        stages = ask(running)
        record.launch(launch_plan, objs, running, capture_submit_provenance())
        with asked(stages):
            _run_in_process(dag, running, launch_plan)
    if not load:
        return []
    # Load the way a cache hit does, so the first run returns what every later
    # load will.
    loaded = _load(objs)
    return [loaded[obj.object_id] for obj in objs]


def _run_in_process(dag: Dag, running: Sequence[Spec], launch_plan: Path) -> None:
    """Create ``running``, the specs of ``dag``, which ``launch_plan`` lists."""
    here = Worker.here()
    firsts: dict[tuple[Worker, str], Spec] = {}
    specs: dict[str, Spec] = {}
    # The plan file that lists each spec, and the batch it joins here.
    plans: dict[str, Path] = {}
    batches: dict[str, tuple[str, int]] = {}
    completed: set[str] = set()
    unseen: dict[str, dict[str, float]] = {}

    def add(new: Sequence[Spec], listed: Path) -> None:
        for obj in new:
            specs[obj.object_id], plans[obj.object_id] = obj, listed
            # A batch with two Metadata stops the launch here.
            batches[obj.object_id] = batch_of(obj, here, firsts)

    add(running, launch_plan)
    while dag.ready:
        lead = next(iter(dag.ready))
        # A batch holds specs from one plan, which its run names.
        group = list(
            islice(
                (
                    spec
                    for spec in dag.ready
                    if batches[spec] == batches[lead] and plans[spec] == plans[lead]
                ),
                batches[lead][1],
            )
        )
        for spec in group:
            del dag.ready[spec]
        objs = [specs[spec] for spec in group]

        try:
            with plan.listed_in(plans[lead]):
                _create_in_process(objs)
        except _DependencyNotReady as exc:
            label = _group_label(objs[0]._log_label, len(objs))
            objs[0].logger.info(
                "blocked %s · %s", label, missing_dependencies(len(exc.dependencies))
            )
            unbuilt = [
                dep
                for dep in exc.dependencies
                if dep.object_id not in completed and not dep.done
            ]
            if unbuilt:
                add(missing_under(unbuilt, dag), exc.plan)
            found = [dep.object_id for dep in exc.dependencies]
            for spec in group:
                # In one process a built spec that create() found missing can
                # only be a race with another run publishing it, which one
                # rerun settles.
                why = why_stuck(unseen.setdefault(spec, {}), found, bool(unbuilt), 0.0)
                if why is not None:
                    raise RuntimeError(f"{specs[spec]._log_label} {why}") from None
                dag.block(spec, [dep.object_id for dep in unbuilt])
            continue
        except Exception as exc:
            if dependents := dag.waiting_for([lead])[:1]:
                exc.add_note(
                    "while building dependencies of " + specs[dependents[0]]._log_label
                )
            raise
        for spec in group:
            dag.complete(spec)
        completed.update(group)


def _create_in_process[T](group: list[Spec[T]]) -> None:
    """Create one group in this process, announced on the terminal."""
    lead = group[0]
    label = _group_label(lead._log_label, len(group))
    lead.logger.info(
        "creating %s", label, extra={"path": log_path_in(working_folder(group))}
    )
    started_at = time.monotonic()
    _ensure_group_result(group)
    duration = format_duration(time.monotonic() - started_at)
    lead.logger.info("finished %s ok · %s", label, duration)


def _create_and_store_group[T](group: list[Spec[T]], *, has_lock: HasLock) -> None:
    """Run the group's create() in its one working folder, then publish it.

    events.jsonl gets this attempt's start and end, and what create() writes
    goes to the folder's log: one call, one folder, one log. The run is added
    to this process's record before it is published, so every published run
    has a line there.
    """
    lead = group[0]
    attempt = working_folder(group)
    attempt.mkdir(exist_ok=True)
    with _log_scope(log_path_in(attempt)):
        resumed = _open_attempt(group, attempt)
        started = events.start(
            attempt,
            [obj._artifact_hash for obj in group],
            plan.listing(),
            record.own().path,
        )
        try:
            results, run, loaded = _call_create(group, attempt, resumed)
            trace = _store(group, results, attempt, run=run, has_lock=has_lock)
        except _DependencyNotReady as exc:
            # Without the lock the folder is a successor's, or published.
            if has_lock():
                waiting_on = [dep.object_id for dep in exc.dependencies]
                events.end(attempt, "blocked", waiting_on=waiting_on)
            raise
        except BaseException as exc:
            if isinstance(exc, Exception):
                lead.logger.exception(
                    "create failed for %s",
                    _group_label(lead._log_label, len(group)),
                )
            if has_lock():
                error = ": ".join([type(exc).__name__, *str(exc).splitlines()[:1]])
                events.end(attempt, "failed", error=error)
            raise
        ended = events.end(attempt, "ok")
        record.batch(
            group,
            run=attempt.name,
            version=FIXED if trace is None else trace.version,
            start=started,
            end=ended,
            loaded=loaded,
        )
        _publish(group, trace, has_lock=has_lock)
