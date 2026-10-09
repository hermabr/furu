from __future__ import annotations

import json
import shutil
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    overload,
)

from furu._batched import _BatchedHook
from furu._declared_types import declared_result_type
from furu._tree import map_specs, specs_in
from furu.config import get_config
from furu.core import Missing, Spec
from furu.dependencies import (
    collect_declared_refs,
    dependency_recorder,
    record_dependency_call,
    under_creation,
)
from furu.locking import lock
from furu.logging import (
    _close_sections,
    _open_sections,
    _run_log_scope,
    get_logger,
)
from furu.metadata import RunningMetadata
from furu.migration.links import load_stored_result, result_dir_for_loading
from furu.migration.stale import raise_if_stale
from furu.provenance import (
    ExecuteContext,
    Provenance,
    SubmitProvenance,
    _require_uv,
    capture_submit_provenance,
)
from furu.resources import Worker
from furu.result.bundle import _save_result_bundle, load_result_bundle
from furu.storage._layout import (
    compute_lock_path_in,
    data_dir_in,
    metadata_path_in,
    provenance_path_in,
    result_dir_in,
    result_link_path_in,
    run_log_path_in,
    schema_snapshot_path_in,
    scratch_dir_in,
)
from furu.utils import (
    atomic_write_text,
    error_summary,
    format_duration,
    nfs_safe_unique_name,
)
from furu.worker.context import (
    _DependencyNotReady,
    _in_worker_execution,
)

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


def _store_result[T](
    obj: Spec[T],
    result: T,
    *,
    metadata: RunningMetadata,
    observed_dependencies: tuple[str, ...],
    has_lock: HasLock,
    submit_provenance: SubmitProvenance,
) -> T:
    lock_path = compute_lock_path_in(obj._base_dir)
    result_dir = result_dir_in(obj._base_dir)
    if not has_lock():
        raise RuntimeError(f"lost lock at {lock_path} before writing final result")

    tmp_result_dir = nfs_safe_unique_name(result_dir, name="tmp")

    declared_type = declared_result_type(type(obj))
    data_dir = data_dir_in(obj._base_dir)

    dump_state = _save_result_bundle(
        result,
        tmp_result_dir,
        declared_type=declared_type,
        result_codecs=obj.result_codecs,
        data_dir=data_dir,
    )

    if not has_lock():
        raise RuntimeError(f"lost lock at {lock_path} after writing temporary result")

    tmp_result_dir.rename(result_dir)
    result_link_path_in(obj._base_dir).unlink(missing_ok=True)

    _record_schema_snapshot(obj)

    metadata_text = metadata.to_complete(
        observed_dependencies=observed_dependencies
    ).model_dump_json(indent=2)
    atomic_write_text(metadata_path_in(obj._base_dir), metadata_text)

    provenance = Provenance.merge(submit_provenance, ExecuteContext.capture())
    atomic_write_text(
        provenance_path_in(obj._base_dir), provenance.model_dump_json(indent=2)
    )

    for binding in dump_state.ref_bindings:
        binding.ref._bind_stored(
            metadata=binding.metadata,
            artifact_directory=result_dir / binding.artifact_relative_path,
        )

    if dump_state.should_reload_value_after_save:
        return cast(
            T,
            load_result_bundle(
                result_dir, data_dir=data_dir, declared_type=declared_type
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
    for obj in objs:
        record_dependency_call(obj)
    if _in_worker_execution.get():
        outputs = _load_or_create_worker(objs, load=load)
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
        if result_dir_for_loading(obj) is not None:
            obj.logger.info("cache hit for %s", obj._log_label)
            continue
        raise_if_stale(obj)
        obj._base_dir.mkdir(parents=True, exist_ok=True)
        missing.append(obj)

    if not missing:
        return

    with lock([compute_lock_path_in(obj._base_dir) for obj in missing]) as has_lock:
        pending = [
            obj for obj in missing if result_dir_for_loading(obj, has_lock=True) is None
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
        record_dependency_call(obj)
        if (result_dir := result_dir_for_loading(obj)) is None:
            raise_if_stale(obj)
            missing.append(obj)
            continue
        loaded[obj.object_id] = load_stored_result(obj, result_dir)
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


def _load_or_create_worker[T](objs: list[Spec[T]], *, load: bool) -> list[T]:
    loaded: list[T] = []
    cached: list[Spec[T]] = []
    missing: list[Spec[T]] = []

    for obj in objs:
        if (cached_result_dir := result_dir_for_loading(obj)) is not None:
            if load:
                loaded.append(load_stored_result(obj, cached_result_dir))
            cached.append(obj)
        else:
            raise_if_stale(obj)
            missing.append(obj)

    if cached:
        objs[0].logger.info("%s", _cached_to_build_msg(cached, missing))

    if missing:
        raise _DependencyNotReady(dependencies=missing)

    return loaded


def _load_or_create_local[T](
    objs: list[Spec[T]],
    *,
    dependents: Sequence[Spec] = (),
    load: bool = True,
) -> list[T]:
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
        if (cached_result_dir := result_dir_for_loading(obj)) is not None:
            cached.append(obj)
            if load:
                results_by_object_id[obj.object_id] = load_stored_result(
                    obj, cached_result_dir
                )
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

    # Declared dependencies are built first, matching the coordinator's DAG.
    _load_or_create_local(
        [ref for obj in missing for ref in collect_declared_refs(obj)],
        dependents=missing,
        load=False,
    )
    for obj in missing:
        obj._base_dir.mkdir(parents=True, exist_ok=True)

    lock_ctx = (
        lock([compute_lock_path_in(obj._base_dir) for obj in missing])
        if missing
        else nullcontext(lambda: True)
    )

    with lock_ctx as has_lock:
        pending: list[Spec[T]] = []
        late_hits = 0
        for obj in missing:
            if (
                cached_result_dir := result_dir_for_loading(obj, has_lock=True)
            ) is not None:
                late_hits += 1
                if load:
                    results_by_object_id[obj.object_id] = load_stored_result(
                        obj, cached_result_dir
                    )
            else:
                pending.append(obj)

        if late_hits:
            objs[0].logger.info(
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

    if not load:
        return []
    return [results_by_object_id[obj.object_id] for obj in objs]


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
    # Inside another create(), this line lands in that one's run.log.
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
    metadata = [RunningMetadata.write_for(obj) for obj in group]

    try:
        match getattr(type(group[0]), "_furu_create_hook", None):
            case None:
                raise TypeError(
                    f"{type(group[0]).__qualname__} cannot create missing results "
                    "because it does not define create()"
                )
            case _BatchedHook(func=create_hook):
                with dependency_recorder() as recorder, under_creation(group):
                    results = create_hook(group)
                observed = recorder.finalize()
                if not isinstance(results, list):
                    raise TypeError(
                        f"{type(group[0]).__name__}.create() must return a list"
                    )
                # TODO: Track dependency calls per object during batched execution.
                # This currently assigns dependencies observed anywhere in the batch
                # to every object.
                observed_dependencies = [observed for _ in group]
            case create_hook:
                results = []
                observed_dependencies = []
                for obj in group:
                    with dependency_recorder() as recorder, under_creation([obj]):
                        results.append(create_hook(obj))
                    observed_dependencies.append(recorder.finalize())

        if len(results) != len(group):
            raise TypeError(
                f"{type(group[0]).__name__} returned {len(results)} results for {len(group)} objects"
            )

        for obj, result, observed_dependency_ids, obj_metadata in zip(
            group,
            results,
            observed_dependencies,
            metadata,
            strict=True,
        ):
            results_by_object_id[obj.object_id] = _store_result(
                obj,
                result,
                metadata=obj_metadata,
                observed_dependencies=observed_dependency_ids,
                has_lock=has_lock,
                submit_provenance=submit_provenance,
            )
            shutil.rmtree(scratch_dir_in(obj._base_dir), ignore_errors=True)
    except Exception:
        group[0].logger.exception("create failed for %s", group[0]._log_label)
        raise
