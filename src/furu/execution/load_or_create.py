from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    overload,
)

from furu._batched import _BatchedHook
from furu._declared_types import declared_result_type
from furu._pytree import flatten_specs
from furu.config import get_config
from furu.core import Missing, Spec
from furu.dependencies import (
    dependency_recorder,
    record_dependency_call,
    under_creation,
)
from furu.locking import lock
from furu.logging import _scoped_log_files, get_logger
from furu.metadata import RunningMetadata
from furu.migration.links import result_dir_for_loading
from furu.migration.stale import raise_if_stale
from furu.provenance import (
    ExecuteContext,
    Provenance,
    SubmitProvenance,
    _require_uv,
    capture_submit_provenance,
)
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
from furu.utils import atomic_write_text, format_duration, nfs_safe_unique_name
from furu.worker.context import (
    _DependencyNotReady,
    _in_worker_execution,
)

if TYPE_CHECKING:
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

    obj.logger.debug("stored result bundle at %s", result_dir)

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


def _load_or_create(tree: object, *, use_lock: bool = True) -> Any:
    """Load or create every Spec leaf of ``tree``; return results in the same shape."""
    _require_uv()
    leaves, unflatten = flatten_specs(tree)
    for obj in leaves:
        record_dependency_call(obj)
    if len(leaves) == 1:
        leaves[0].logger.debug(".create called for %s", leaves[0])
    if _in_worker_execution.get():
        return unflatten(_load_or_create_worker(leaves))
    return unflatten(_load_or_create_local(leaves, use_lock=use_lock))


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


@overload
def create[T](obj: Spec[T], *, on: Sequence[WorkerBackend] | None = None) -> T: ...
@overload
def create[T](
    objs: tuple[Spec[T], ...], *, on: Sequence[WorkerBackend] | None = None
) -> tuple[T, ...]: ...
@overload
def create[T](
    objs: Sequence[Spec[T]], *, on: Sequence[WorkerBackend] | None = None
) -> list[T]: ...
@overload
def create(tree: object, *, on: Sequence[WorkerBackend] | None = None) -> Any: ...
def create(tree: object, *, on: Sequence[WorkerBackend] | None = None) -> Any:
    """Load or create a pytree of Specs, returning results in the same shape."""
    if on is not None:
        from furu.execution.execution_coordinator import ExecutionCoordinator

        ExecutionCoordinator.run(flatten_specs(tree)[0], worker_backends=tuple(on))
    return _load_or_create(tree)


@overload
def load_existing[T](obj: Spec[T]) -> T: ...
@overload
def load_existing[T](objs: tuple[Spec[T], ...]) -> tuple[T, ...]: ...
@overload
def load_existing[T](objs: Sequence[Spec[T]]) -> list[T]: ...
@overload
def load_existing(tree: object) -> Any: ...
def load_existing(tree: object) -> Any:
    """Load existing results for a pytree of Specs, returning the same shape."""
    objs, unflatten = flatten_specs(tree)
    loaded: list[Any] = []
    missing: list[Spec] = []
    for obj in objs:
        record_dependency_call(obj)
        if (result_dir := result_dir_for_loading(obj)) is None:
            raise_if_stale(obj)
            missing.append(obj)
            continue
        loaded.append(
            load_result_bundle(
                result_dir,
                data_dir=data_dir_in(result_dir.parent),
                declared_type=declared_result_type(type(obj)),
            )
        )
    if missing:
        if _in_worker_execution.get():
            raise _DependencyNotReady(dependencies=missing, call_kind="load_existing")
        first = missing[0]
        raise Missing(
            f"{first._log_label}.load_existing() could not find a result. "
            "load_existing() only loads existing results; use create() to compute "
            "missing results."
        )
    if objs:
        get_logger().info(
            "loaded %d furu objects including %s", len(loaded), objs[0]._log_label
        )
    else:
        get_logger().info("loaded 0 furu objects")
    return unflatten(loaded)


def _cached_to_build_msg(cached: list[Spec], to_build: list[Spec]) -> str:
    def fmt(objs: list[Spec]) -> str:
        if len(cached) + len(to_build) > 5:
            return str(len(objs))
        return ", ".join(o._log_label for o in objs)

    msg = f"cached {fmt(cached)}"
    return f"building {fmt(to_build)}, {msg}" if to_build else msg


def _load_or_create_worker[T](objs: list[Spec[T]]) -> list[T]:
    loaded: list[T] = []
    cached: list[Spec[T]] = []
    missing: list[Spec[T]] = []

    for obj in objs:
        if (cached_result_dir := result_dir_for_loading(obj)) is not None:
            loaded.append(
                cast(
                    T,
                    load_result_bundle(
                        cached_result_dir,
                        data_dir=data_dir_in(cached_result_dir.parent),
                        declared_type=declared_result_type(type(obj)),
                    ),
                )
            )
            cached.append(obj)
        else:
            raise_if_stale(obj)
            missing.append(obj)

    if loaded:
        objs[0].logger.info("%s", _cached_to_build_msg(cached, missing))

    if missing:
        raise _DependencyNotReady(
            dependencies=missing,
            call_kind="create",
        )

    return loaded


def _load_or_create_local[T](objs: list[Spec[T]], *, use_lock: bool = True) -> list[T]:
    if not objs:
        return []

    unique_by_object_id: dict[str, Spec[T]] = {}
    for obj in objs:
        unique_by_object_id.setdefault(obj.object_id, obj)
    unique = list(unique_by_object_id.values())

    results_by_object_id: dict[str, T] = {}
    missing: list[Spec[T]] = []

    for obj in unique:
        if (cached_result_dir := result_dir_for_loading(obj)) is not None:
            results_by_object_id[obj.object_id] = cast(
                T,
                load_result_bundle(
                    cached_result_dir,
                    data_dir=data_dir_in(cached_result_dir.parent),
                    declared_type=declared_result_type(type(obj)),
                ),
            )
        else:
            raise_if_stale(obj)
            obj._base_dir.mkdir(parents=True, exist_ok=True)
            missing.append(obj)

    if results_by_object_id:
        cached = [o for o in unique if o.object_id in results_by_object_id]
        unique[0].logger.info("%s", _cached_to_build_msg(cached, missing))

    lock_ctx = (
        lock([compute_lock_path_in(obj._base_dir) for obj in missing])
        if use_lock and missing
        else nullcontext()
    )

    with lock_ctx as maybe_has_lock:
        has_lock = maybe_has_lock or (lambda: True)
        pending: list[Spec[T]] = []
        late_hits = 0
        for obj in missing:
            if (
                cached_result_dir := result_dir_for_loading(obj, has_lock=use_lock)
            ) is not None:
                late_hits += 1
                results_by_object_id[obj.object_id] = cast(
                    T,
                    load_result_bundle(
                        cached_result_dir,
                        data_dir=data_dir_in(cached_result_dir.parent),
                        declared_type=declared_result_type(type(obj)),
                    ),
                )
            else:
                pending.append(obj)

        if late_hits:
            objs[0].logger.info(
                "%d became ready while waiting, %d to build", late_hits, len(pending)
            )

        direct_create_started = len(objs) == 1 and bool(pending)
        create_started_at = time.monotonic()
        if direct_create_started:
            objs[0].logger.info("creating %s", objs[0]._log_label)

        if pending:
            submit_provenance = capture_submit_provenance(
                snapshot=get_config().provenance.snapshot
            )

            for group in _grouped_pending(pending):
                _create_and_store_group(
                    group,
                    has_lock=has_lock,
                    results_by_object_id=results_by_object_id,
                    submit_provenance=submit_provenance,
                )

    if direct_create_started:
        objs[0].logger.info(
            "finished %s ok · %s",
            objs[0]._log_label,
            format_duration(time.monotonic() - create_started_at),
        )
    return [results_by_object_id[obj.object_id] for obj in objs]


def _batch_group(obj: Spec) -> tuple[object, int] | None:
    hook = getattr(type(obj), "_furu_create_hook", None)
    if not isinstance(hook, _BatchedHook):
        return None
    group_hash, cap = hook.batch_fn(obj)
    if type(cap) is not int or cap < 1:
        raise TypeError(
            f"{type(obj).__qualname__} batch key cap must be a positive int, "
            f"got {cap!r}"
        )
    key = (type(obj), group_hash, cap, obj._metadata.requires, obj._metadata.execution)
    return key, cap


def _grouped_pending[T](pending: list[Spec[T]]) -> list[list[Spec[T]]]:
    """Partition by (type, batch_key, requires, execution), chunked to the cap."""
    groups: list[tuple[object, int | None, list[Spec[T]]]] = []
    for obj in pending:
        key, cap = _batch_group(obj) or (type(obj), None)
        for existing_key, _, group in groups:
            if existing_key == key:
                group.append(obj)
                break
        else:
            groups.append((key, cap, [obj]))
    return [
        group[i : i + (cap or len(group))]
        for _, cap, group in groups
        for i in range(0, len(group), cap or len(group))
    ]


def _create_and_store_group[T](
    group: list[Spec[T]],
    *,
    has_lock: HasLock,
    results_by_object_id: dict[str, T],
    submit_provenance: SubmitProvenance,
) -> None:
    log_paths = tuple(run_log_path_in(obj._base_dir) for obj in group)

    metadata = [RunningMetadata.write_for(obj) for obj in group]

    with _scoped_log_files(log_paths):
        logger = group[0].logger
        logger.debug("create start")
        group_started_at = time.monotonic()
        try:
            match getattr(type(group[0]), "_furu_create_hook", None):
                case None:
                    raise TypeError(
                        f"{type(group[0]).__qualname__} cannot create missing results "
                        "because it does not define create()"
                    )
                case _BatchedHook(func=create_hook):
                    logger.debug("running batched create() hook")
                    with dependency_recorder() as recorder, under_creation(group):
                        results = create_hook(group)
                    observed = recorder.finalize()
                    logger.debug("batched create() hook returned")
                    if not isinstance(results, list):
                        raise TypeError(
                            f"{type(group[0]).__name__}.create() must return a list"
                        )
                    # TODO: Track dependency calls per object during batched execution.
                    # This currently assigns dependencies observed anywhere in the batch
                    # to every object.
                    observed_dependencies = [observed for _ in group]
                case create_hook:
                    logger.debug("running sequential create() fallback")
                    results = []
                    observed_dependencies = []
                    for obj in group:
                        with dependency_recorder() as recorder, under_creation([obj]):
                            results.append(create_hook(obj))
                        observed_dependencies.append(recorder.finalize())
                    logger.debug("sequential create() fallback returned")

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

            logger.debug(
                "create complete · %s",
                format_duration(time.monotonic() - group_started_at),
            )
        except _DependencyNotReady as exc:
            logger.debug(
                "create deferred: %s discovered %d missing dependency/dependencies",
                exc.call_kind,
                len(exc.dependencies),
            )
            raise
        except Exception:
            logger.exception(
                "create failed for %s", group[0]._log_label, stack_info=True
            )
            raise
