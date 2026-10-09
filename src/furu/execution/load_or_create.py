from __future__ import annotations

import json
import shutil
import time
import traceback
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    overload,
)

from furu._batched import _BatchedHook
from furu._declared_types import declared_result_type
from furu._tree import map_specs, specs_in
from furu.code_trace import (
    LogState,
    Recording,
    build_trace,
    recording,
    why_not_resumable,
    why_outdated,
)
from furu.config import get_config
from furu.core import Missing, Spec
from furu.dependencies import (
    _DependencyNotReady,
    collect_declared_refs,
    dependency_recorder,
    missing_dependencies,
    record_dependency,
    specs_under_creation,
    under_creation,
)
from furu.locking import lock
from furu.logging import (
    _close_sections,
    _open_sections,
    _run_log_scope,
    get_logger,
)
from furu.metadata import ArtifactSpec
from furu.migration.links import load_stored_result, version_for_loading
from furu.migration.stale import raise_if_stale
from furu.provenance import (
    ExecuteContext,
    Provenance,
    SubmitProvenance,
    _require_uv,
    capture_submit_provenance,
)
from furu.resources import Worker
from furu.result.bundle import (
    _DumpState,
    _save_result_bundle,
    bind_refs,
    load_result_bundle,
)
from furu.storage._layout import (
    attempt_dir_in,
    compute_lock_path_in,
    data_dir_in,
    provenance_path_in,
    result_dir_in,
    result_link_path_in,
    run_log_path_in,
    schema_snapshot_path_in,
    scratch_dir_in,
    spec_path_in,
    trace_log_path_in,
    trace_path_in,
)
from furu.utils import atomic_write_text, error_summary, format_duration

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from pydantic import BaseModel

    from furu.worker.backends.protocol import WorkerBackend

type HasLock = Callable[[], bool]


def _record_schema_snapshot(obj: Spec) -> None:
    schema_path = schema_snapshot_path_in(obj._base_dir)
    if schema_path.exists():
        return
    atomic_write_text(
        schema_path, json.dumps(obj._schema_data, indent=2, sort_keys=True)
    )


def _prepare_attempt(obj: Spec) -> LogState:
    """Write spec.json and decide whether attempt/ from an earlier try resumes.

    Fixed specs always resume. A traced spec resumes only if the code its
    earlier attempts ran (recorded in trace.log) is unchanged on disk.
    """
    if not (spec_path := spec_path_in(obj._base_dir)).exists():
        atomic_write_text(
            spec_path, ArtifactSpec.from_furu(obj).model_dump_json(indent=2)
        )
    metadata = obj._metadata
    if why := why_outdated(
        obj._base_dir, metadata.code_version, storage_root=metadata.storage
    ):
        obj.logger.info("rerunning %s: %s", obj._log_label, why)
    attempt = attempt_dir_in(obj._base_dir)
    resumed = LogState()
    if metadata.code_version == "traced" and attempt.exists():
        logged = LogState.read(trace_log_path_in(attempt))
        if why := why_not_resumable(logged, storage_root=metadata.storage):
            obj.logger.info("starting %s over: %s", obj._log_label, why)
            shutil.rmtree(attempt)
        else:
            assert logged is not None
            resumed = logged
    attempt.mkdir(exist_ok=True)
    return resumed


@contextmanager
def _recording(objs: Sequence[Spec], resumed: LogState) -> Generator[Recording]:
    """Trace one create() call and record the dependency versions it loads."""
    metadata = objs[0]._metadata  # a batch shares its metadata
    log_paths = [trace_log_path_in(attempt_dir_in(obj._base_dir)) for obj in objs]
    with (
        recording(
            log_paths, code_version=metadata.code_version, resumed=resumed
        ) as run,
        dependency_recorder(metadata.storage, run.dependencies, run.log),
        under_creation(objs),
    ):
        yield run


def _store_result[T](
    obj: Spec[T],
    result: T,
    *,
    run: Recording,
    has_lock: HasLock,
    submit_provenance: SubmitProvenance,
    started_at: datetime,
) -> tuple[str, _DumpState]:
    """Store the result and its trace in attempt/; return the version name."""
    attempt = attempt_dir_in(obj._base_dir)
    lock_path = compute_lock_path_in(obj._base_dir)
    if not has_lock():
        raise RuntimeError(f"lost lock at {lock_path} before writing final result")

    result_dir = result_dir_in(attempt)
    shutil.rmtree(result_dir, ignore_errors=True)  # left by a crash mid-publish
    dump_state = _save_result_bundle(
        result,
        result_dir,
        declared_type=declared_result_type(type(obj)),
        result_codecs=obj.result_codecs,
        data_dir=data_dir_in(attempt),
    )

    trace = build_trace(run, label=obj._log_label, warn=obj.logger.warning)
    atomic_write_text(
        trace_path_in(attempt), trace.model_dump_json(indent=2, exclude_none=True)
    )
    provenance = Provenance.merge(
        submit_provenance, ExecuteContext.capture(started_at=started_at)
    )
    atomic_write_text(provenance_path_in(attempt), provenance.model_dump_json(indent=2))
    _record_schema_snapshot(obj)

    version_name = trace.version_name()
    obj.logger.debug("stored %s as %s", obj._log_label, version_name)
    return version_name, dump_state


def _publish[T](
    obj: Spec[T],
    result: T,
    *,
    version_name: str,
    dump_state: _DumpState,
    has_lock: HasLock,
) -> T:
    """Rename attempt/ to its version directory; return the value to hand out."""
    attempt = attempt_dir_in(obj._base_dir)
    shutil.rmtree(scratch_dir_in(attempt), ignore_errors=True)
    trace_log_path_in(attempt).unlink(missing_ok=True)
    if not has_lock():
        raise RuntimeError(
            f"lost lock at {compute_lock_path_in(obj._base_dir)} before publishing"
        )
    version = obj._base_dir / version_name
    reload = dump_state.should_reload_value_after_save
    try:
        attempt.rename(version)
    except OSError:
        if not version.is_dir():
            raise
        shutil.rmtree(attempt)  # the same version was published first; keep it
        reload = True
    result_link_path_in(obj._base_dir).unlink(missing_ok=True)

    declared_type = declared_result_type(type(obj))
    bind_refs(dump_state, result_dir_in(version), data_dir=data_dir_in(version))
    if reload:
        return cast(
            T,
            load_result_bundle(
                result_dir_in(version),
                data_dir=data_dir_in(version),
                declared_type=declared_type,
            ),
        )
    return result


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
        from furu.execution.execution_coordinator import ExecutionCoordinator

        ExecutionCoordinator.run(specs_in(tree), worker_backends=tuple(on))
    _require_uv()
    if isinstance(tree, Spec):
        tree.logger.debug(".create called for %s", tree)
    objs = specs_in(tree)
    if creating := specs_under_creation():
        outputs = _load_or_block(objs, load=load, dependents=creating)
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
        if version_for_loading(obj) is not None:
            obj.logger.info("cache hit for %s", obj._log_label)
            continue
        raise_if_stale(obj)
        obj._base_dir.mkdir(parents=True, exist_ok=True)
        missing.append(obj)

    if not missing:
        return

    with lock([compute_lock_path_in(obj._base_dir) for obj in missing]) as has_lock:
        pending = [
            obj for obj in missing if version_for_loading(obj, has_lock=True) is None
        ]
        if pending:
            _create_and_store_group(
                pending,
                has_lock=has_lock,
                results_by_object_id={},
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
        if (version := version_for_loading(obj)) is None:
            raise_if_stale(obj)
            missing.append(obj)
            continue
        record_dependency(obj, version)
        loaded[obj.object_id] = load_stored_result(obj, version)
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


def _load_or_block[T](
    objs: list[Spec[T]], *, load: bool, dependents: Sequence[Spec]
) -> list[T]:
    """Load ``objs`` inside ``dependents``' create hook, or block it if missing.

    Blocking aborts the hook; the caller builds the missing specs and reruns it
    from scratch, so specs never build inside another spec's create.
    """
    loaded: list[T] = []
    cached: list[Spec[T]] = []
    missing: list[Spec[T]] = []

    for obj in objs:
        if (version := version_for_loading(obj)) is not None:
            record_dependency(obj, version)
            if load:
                loaded.append(load_stored_result(obj, version))
            cached.append(obj)
        else:
            raise_if_stale(obj)
            missing.append(obj)

    if cached:
        objs[0].logger.info("%s", _cached_to_build_msg(cached, missing))

    if missing:
        raise _DependencyNotReady(dependencies=missing, dependents=dependents)

    return loaded


def _load_or_create_local[T](
    objs: list[Spec[T]],
    *,
    load: bool = True,
    dependents: Sequence[Spec] = (),
    waiting: tuple[Spec, ...] = (),
) -> list[T]:
    """Load or build ``objs`` in this process.

    ``dependents`` are the specs this call builds for and ``waiting`` the ones
    further up the stack; meeting one of them again is a dependency cycle. A
    create that blocks on missing specs is aborted with its locks released,
    they are built, and it reruns from scratch, like on a worker.
    """
    if not objs:
        return []

    unique_by_object_id: dict[str, Spec[T]] = {}
    for obj in objs:
        unique_by_object_id.setdefault(obj.object_id, obj)
    unique = list(unique_by_object_id.values())

    results_by_object_id: dict[str, T] = {}
    cached: list[Spec[T]] = []
    missing: list[Spec[T]] = []

    for obj in unique:
        if (version := version_for_loading(obj)) is not None:
            cached.append(obj)
            if load:
                results_by_object_id[obj.object_id] = load_stored_result(obj, version)
        else:
            raise_if_stale(obj)
            missing.append(obj)

    if cached:
        unique[0].logger.info("%s", _cached_to_build_msg(cached, missing))

    if dependents and missing:
        others = f" and {len(dependents) - 1} others" if len(dependents) > 1 else ""
        dependents[0].logger.info(
            "building %d %s of %s%s",
            len(missing),
            "dependency" if len(missing) == 1 else "dependencies",
            dependents[0]._log_label,
            others,
        )

    waiting = (*waiting, *dependents)
    waiting_ids = [spec.object_id for spec in waiting]
    for obj in missing:
        if obj.object_id in waiting_ids:
            cycle = (*waiting[waiting_ids.index(obj.object_id) :], obj)
            raise RuntimeError(
                "dependency cycle: " + " → ".join(spec._log_label for spec in cycle)
            )

    # Declared dependencies are built first, matching the coordinator's DAG.
    _load_or_create_local(
        [ref for obj in missing for ref in collect_declared_refs(obj)],
        load=False,
        dependents=missing,
        waiting=waiting,
    )
    for obj in missing:
        obj._base_dir.mkdir(parents=True, exist_ok=True)

    built: set[str] = set()
    while missing:
        try:
            _create_missing(missing, results_by_object_id, load=load)
            break
        except _DependencyNotReady as exc:
            blocked = exc
        # The locks are released: build what the create asked for, then rerun.
        parent = blocked.dependents[0]._log_label
        for dep in blocked.dependencies:
            if dep.object_id in built:
                raise RuntimeError(
                    f"{parent} still finds {dep._log_label} missing after building it"
                )
        try:
            _load_or_create_local(
                list(blocked.dependencies),
                load=False,
                dependents=blocked.dependents,
                waiting=waiting,
            )
        except Exception as exc:
            exc.add_note(f"while building dependencies of {parent}")
            raise
        built.update(dep.object_id for dep in blocked.dependencies)

    if not load:
        return []
    return [results_by_object_id[obj.object_id] for obj in objs]


def _create_missing[T](
    missing: list[Spec[T]], results_by_object_id: dict[str, T], *, load: bool
) -> None:
    with lock([compute_lock_path_in(obj._base_dir) for obj in missing]) as has_lock:
        pending: list[Spec[T]] = []
        late_hits = 0
        for obj in missing:
            if (version := version_for_loading(obj, has_lock=True)) is not None:
                late_hits += 1
                if load:
                    results_by_object_id[obj.object_id] = load_stored_result(
                        obj, version
                    )
            else:
                pending.append(obj)

        if late_hits:
            missing[0].logger.info(
                "%d became ready while waiting, %d to build", late_hits, len(pending)
            )

        if pending:
            submit_provenance = capture_submit_provenance(
                snapshot=get_config().provenance.snapshot
            )

            for group in _grouped_pending(pending):
                _create_in_process(
                    group,
                    has_lock=has_lock,
                    results_by_object_id=results_by_object_id if load else {},
                    submit_provenance=submit_provenance,
                )


def _create_in_process[T](
    group: list[Spec[T]],
    *,
    has_lock: HasLock,
    results_by_object_id: dict[str, T],
    submit_provenance: SubmitProvenance,
) -> None:
    """Create one group in this process, announced on the terminal.

    The lead's run.log gets this attempt's log records, framed by a header and
    a footer; the other members' run.logs point at it.
    """
    lead = group[0]
    label = lead._log_label + (f" ×{len(group)}" if len(group) > 1 else "")
    run_logs = [(obj._log_label, run_log_path_in(obj._base_dir)) for obj in group]
    lead.logger.info("creating %s", label, extra={"path": run_logs[0][1]})
    started_at = time.monotonic()
    with _run_log_scope(run_logs[0][1]) as run_log:
        _open_sections(run_logs, run_log, "in-process")
        try:
            _create_and_store_group(
                group,
                has_lock=has_lock,
                results_by_object_id=results_by_object_id,
                submit_provenance=submit_provenance,
            )
        except _DependencyNotReady as exc:
            duration = format_duration(time.monotonic() - started_at)
            missing = missing_dependencies(len(exc.dependencies))
            lead.logger.info("blocked %s · %s", label, missing)
            _close_sections(run_logs, run_log, f"blocked · {duration} · {missing}")
            raise
        except BaseException as exc:
            duration = format_duration(time.monotonic() - started_at)
            summary = error_summary("".join(traceback.format_exception_only(exc)))
            _close_sections(run_logs, run_log, f"failed · {duration} · {summary}")
            raise
        duration = format_duration(time.monotonic() - started_at)
        _close_sections(run_logs, run_log, f"ok · {duration}")
    lead.logger.info("finished %s ok · %s", label, duration)


def _batch_group(obj: Spec, worker: Worker) -> tuple[object, int] | None:
    hook = getattr(type(obj), "_furu_create_hook", None)
    if not isinstance(hook, _BatchedHook):
        return None
    group_hash, cap = hook.batch_fn(obj, worker)
    if type(cap) is not int or cap < 1:
        raise TypeError(
            f"{type(obj).__qualname__} batch key cap must be a positive int, "
            f"got {cap!r} on {worker}"
        )
    key = (type(obj), group_hash, cap, obj._metadata)
    return key, cap


def _grouped_pending[T](pending: list[Spec[T]]) -> list[list[Spec[T]]]:
    """Partition by (type, batch_key, metadata), chunked to the cap.

    Unbatched specs run alone, so each is stored as soon as it finishes.
    """
    here = Worker.here()
    groups: list[tuple[object, int, list[Spec[T]]]] = []
    for obj in pending:
        key, cap = _batch_group(obj, here) or (type(obj), 1)
        for existing_key, _, group in groups:
            if existing_key == key:
                group.append(obj)
                break
        else:
            groups.append((key, cap, [obj]))
    return [
        group[i : i + cap]
        for _, cap, group in groups
        for i in range(0, len(group), cap)
    ]


def _create_and_store_group[T](
    group: list[Spec[T]],
    *,
    has_lock: HasLock,
    results_by_object_id: dict[str, T],
    submit_provenance: SubmitProvenance,
) -> None:
    started_at = datetime.now(UTC)
    resumed = [_prepare_attempt(obj) for obj in group]
    try:
        match getattr(type(group[0]), "_furu_create_hook", None):
            case None:
                raise TypeError(
                    f"{type(group[0]).__qualname__} cannot create missing results "
                    "because it does not define create()"
                )
            case _BatchedHook(func=create_hook):
                union = LogState()
                for state in resumed:
                    union.merge(state)
                with _recording(group, union) as run:
                    results = create_hook(group)
                if not isinstance(results, list):
                    raise TypeError(
                        f"{type(group[0]).__name__}.create() must return a list"
                    )
                # TODO: Trace code and dependencies per object during batched
                # execution. Every object currently shares the batch's trace.
                runs = [run for _ in group]
            case create_hook:
                results = []
                runs = []
                for obj, state in zip(group, resumed, strict=True):
                    with _recording([obj], state) as run:
                        results.append(create_hook(obj))
                    runs.append(run)

        if len(results) != len(group):
            raise TypeError(
                f"{type(group[0]).__name__} returned {len(results)} results for {len(group)} objects"
            )

        for obj, result, run in zip(group, results, runs, strict=True):
            version_name, dump_state = _store_result(
                obj,
                result,
                run=run,
                has_lock=has_lock,
                submit_provenance=submit_provenance,
                started_at=started_at,
            )
            results_by_object_id[obj.object_id] = _publish(
                obj,
                result,
                version_name=version_name,
                dump_state=dump_state,
                has_lock=has_lock,
            )
    except Exception:
        group[0].logger.exception("create failed for %s", group[0]._log_label)
        raise
