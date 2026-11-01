from __future__ import annotations

import json
import shutil
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, overload

from furu._batched import _BatchedHook
from furu._declared_types import declared_result_type
from furu._tree import map_specs, specs_in
from furu.core import Missing, Spec
from furu.dag import Dag
from furu.dependencies import (
    _DependencyNotReady,
    any_under_creation,
    dependency_recorder,
    missing_dependencies,
    record_dependency,
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
from furu.result.bundle import _save_result_bundle
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


def _publish[T](
    obj: Spec[T],
    result: T,
    *,
    dependencies: Mapping[str, str],
    has_lock: HasLock,
    submit_provenance: SubmitProvenance,
    started_at: datetime,
) -> None:
    """Store the result in attempt/, then publish it by renaming it to v-fixed/."""
    attempt = attempt_dir_in(obj._base_dir)
    lock_path = compute_lock_path_in(obj._base_dir)
    if not has_lock():
        raise RuntimeError(f"lost lock at {lock_path} before writing final result")

    result_dir = result_dir_in(attempt)
    shutil.rmtree(result_dir, ignore_errors=True)  # left by a crash mid-publish
    _save_result_bundle(
        result,
        result_dir,
        declared_type=declared_result_type(type(obj)),
        result_codecs=obj.result_codecs,
        data_dir=data_dir_in(attempt),
    )
    atomic_write_text(
        trace_path_in(attempt),
        json.dumps({"dependencies": dict(sorted(dependencies.items()))}, indent=2),
    )
    provenance = Provenance.merge(
        submit_provenance, ExecuteContext.capture(started_at=started_at)
    )
    atomic_write_text(provenance_path_in(attempt), provenance.model_dump_json(indent=2))
    _record_schema_snapshot(obj)
    shutil.rmtree(scratch_dir_in(attempt), ignore_errors=True)

    if not has_lock():
        raise RuntimeError(f"lost lock at {lock_path} before publishing")
    version = obj._base_dir / "v-fixed"
    try:
        attempt.rename(version)
    except OSError:
        if not version.is_dir():
            raise
        shutil.rmtree(attempt)  # the same version was published first; keep it
    result_link_path_in(obj._base_dir).unlink(missing_ok=True)


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


def _load_or_block[T](objs: list[Spec[T]], *, load: bool) -> list[T]:
    """Load ``objs`` inside a create hook, or block that create if any are missing.

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
                "blocked %s%s · %s",
                lead.obj._log_label,
                f" ×{len(group)}" if len(group) > 1 else "",
                missing_dependencies(len(exc.dependencies)),
            )
            for dep in exc.dependencies:
                if dep.object_id in dag.completed:
                    raise RuntimeError(
                        f"{lead.obj._log_label} still finds {dep._log_label} "
                        "missing after building it"
                    ) from None
            for node in group:
                dag.block(node, exc.dependencies)
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
    """Create one group in this process, announced on the terminal.

    The lead's run.log gets this attempt's log records, framed by a header and
    a footer; the other members' run.logs point at it.
    """
    lead = group[0]
    label = lead._log_label + (f" ×{len(group)}" if len(group) > 1 else "")
    run_logs = [(obj._log_label, run_log_path_in(obj._base_dir)) for obj in group]
    for obj in group:
        obj._base_dir.mkdir(parents=True, exist_ok=True)
    lead.logger.info("creating %s", label, extra={"path": run_logs[0][1]})
    started_at = time.monotonic()
    with _run_log_scope(run_logs[0][1]) as run_log:
        _open_sections(run_logs, run_log, "in-process")
        try:
            _ensure_group_result(group, submit_provenance=submit_provenance)
        except _DependencyNotReady as exc:
            duration = format_duration(time.monotonic() - started_at)
            missing = missing_dependencies(len(exc.dependencies))
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


def _create_and_store_group[T](
    group: list[Spec[T]],
    *,
    has_lock: HasLock,
    submit_provenance: SubmitProvenance,
) -> None:
    started_at = datetime.now(UTC)
    for obj in group:
        # Keep an attempt/ left by an earlier try so create() can resume it.
        if not (spec_path := spec_path_in(obj._base_dir)).exists():
            atomic_write_text(
                spec_path, ArtifactSpec.from_furu(obj).model_dump_json(indent=2)
            )
        attempt_dir_in(obj._base_dir).mkdir(exist_ok=True)
    try:
        match getattr(type(group[0]), "_furu_create_hook", None):
            case None:
                raise TypeError(
                    f"{type(group[0]).__qualname__} cannot create missing results "
                    "because it does not define create()"
                )
            case _BatchedHook(func=create_hook):
                storage = group[0]._metadata.storage  # a batch shares its metadata
                with dependency_recorder(storage) as recorder, under_creation(group):
                    results = create_hook(group)
                if not isinstance(results, list):
                    raise TypeError(
                        f"{type(group[0]).__name__}.create() must return a list"
                    )
                # One create() call built the whole batch, so every object
                # records the dependencies the batch loaded.
                dependencies = [recorder.versions for _ in group]
            case create_hook:
                results = []
                dependencies = []
                for obj in group:
                    storage = obj._metadata.storage
                    with (
                        dependency_recorder(storage) as recorder,
                        under_creation([obj]),
                    ):
                        results.append(create_hook(obj))
                    dependencies.append(recorder.versions)

        if len(results) != len(group):
            raise TypeError(
                f"{type(group[0]).__name__} returned {len(results)} results for {len(group)} objects"
            )

        for obj, result, obj_dependencies in zip(
            group, results, dependencies, strict=True
        ):
            _publish(
                obj,
                result,
                dependencies=obj_dependencies,
                has_lock=has_lock,
                submit_provenance=submit_provenance,
                started_at=started_at,
            )
    except Exception:
        group[0].logger.exception("create failed for %s", group[0]._log_label)
        raise
