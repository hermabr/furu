from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    assert_never,
    cast,
    overload,
)

from furu._declared_types import declared_result_type
from furu.core import SpecCreateMode, Missing, Spec
from furu.dependencies import dependency_recorder, record_dependency_call
from furu.locking import lock
from furu.logging import _scoped_log_files, get_logger
from furu.metadata import RunningMetadata
from furu.migration import raise_if_stale, result_dir_for_loading
from furu.result.bundle import _save_result_bundle, load_result_bundle
from furu.storage._layout import (
    compute_lock_path_in,
    data_dir_in,
    metadata_path_in,
    result_dir_in,
    run_log_path_in,
    schema_snapshot_path_in,
)
from furu.utils import format_duration, nfs_safe_unique_name
from furu.worker.context import (
    _DependencyNotReady,
    _worker_execution_lease_id,
)

if TYPE_CHECKING:
    from furu.worker.backends.protocol import WorkerBackend

HasLock: TypeAlias = Callable[[], bool]
_DirectCreateTarget: TypeAlias = Spec[Any] | type[Spec[Any]]


_direct_create_target: ContextVar[_DirectCreateTarget | None] = ContextVar(
    "_furu_direct_create_target", default=None
)


@contextmanager
def _allow_direct_create(target: _DirectCreateTarget) -> Iterator[None]:
    token = _direct_create_target.set(target)
    try:
        yield
    finally:
        _direct_create_target.reset(token)


def _install_create_dispatchers[T](cls: type[Spec[T]]) -> None:
    if "create" in cls.__dict__:
        raw_create = cls.__dict__["create"]

        @functools.wraps(raw_create)
        def create_dispatcher(self: Spec[T], *args: Any, **kwargs: Any) -> T:
            if _direct_create_target.get() is self:
                return raw_create(self, *args, **kwargs)
            return _load_or_create(self, *args, **kwargs)

        setattr(cls, "create", create_dispatcher)

    if "create_batched" in cls.__dict__:
        raw_create_batched = cls.__dict__["create_batched"]
        func = raw_create_batched.__func__

        @functools.wraps(func)
        def create_batched_guard(
            owner: type[Spec[T]], *args: Any, **kwargs: Any
        ) -> list[T]:
            target = _direct_create_target.get()
            if not (isinstance(target, type) and issubclass(target, owner)):
                raise RuntimeError(
                    f"{owner.__name__}.create_batched() must not be called directly; "
                    "call .create() on Spec objects instead"
                )
            return func(owner, *args, **kwargs)

        setattr(cls, "create_batched", classmethod(create_batched_guard))


def _resolve_create_mode[T](cls: type[Spec[T]]) -> SpecCreateMode:
    defines_single = False
    defines_batched = False

    for base in cls.__mro__:
        if not issubclass(base, Spec) or base is Spec:
            continue

        if "create" in base.__dict__:
            defines_single = True
        if "create_batched" in base.__dict__:
            if not isinstance(base.__dict__["create_batched"], classmethod):
                raise TypeError(
                    f"{base.__qualname__}.create_batched must be a @classmethod"
                )
            defines_batched = True

    if defines_single and defines_batched:
        raise TypeError(
            f"{cls.__qualname__} must define exactly one of create or create_batched"
        )
    if defines_single:
        return "single"
    if defines_batched:
        return "batched"
    return None


def _record_schema_snapshot(obj: Spec[Any]) -> None:
    schema_path = schema_snapshot_path_in(obj._base_dir)
    if schema_path.exists():
        return
    tmp_path = nfs_safe_unique_name(schema_path, name="tmp")
    tmp_path.write_text(json.dumps(obj._schema_data, indent=2, sort_keys=True))
    tmp_path.rename(schema_path)


def _store_result[T](
    obj: Spec[T],
    result: T,
    *,
    metadata: RunningMetadata,
    observed_dependencies: tuple[str, ...],
    has_lock: HasLock,
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

    _record_schema_snapshot(obj)

    metadata_text = metadata.to_complete(
        observed_dependencies=observed_dependencies
    ).model_dump_json(indent=2)
    metadata_path_in(obj._base_dir).write_text(metadata_text)

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


@overload
def _load_or_create[T](obj: Spec[T], *, use_lock: bool = True) -> T: ...


@overload
def _load_or_create[T](
    objs: Sequence[Spec[T]], *, use_lock: bool = True
) -> list[T]: ...


def _load_or_create[T](
    obj_or_objs: Spec[T] | Sequence[Spec[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    if _worker_execution_lease_id.get() is not None:
        return _load_or_create_worker(obj_or_objs)
    return _load_or_create_local(obj_or_objs, use_lock=use_lock)


def _ensure_single_result[T](obj: Spec[T]) -> None:
    if result_dir_for_loading(obj) is not None:
        obj.logger.info("cache hit for %s", obj._log_label)
        return

    raise_if_stale(obj)
    obj._base_dir.mkdir(parents=True, exist_ok=True)

    with lock(compute_lock_path_in(obj._base_dir)) as has_lock:
        if result_dir_for_loading(obj) is not None:
            obj.logger.info("cache hit for %s", obj._log_label)
            return

        _create_and_store_group(
            [obj],
            has_lock=has_lock,
            results_by_object_id={},
        )


def _normalize_load_or_create_input[T](
    obj_or_objs: Spec[T] | Sequence[Spec[T]],
) -> tuple[list[Spec[T]], bool]:
    if isinstance(obj_or_objs, Spec):
        objs = [obj_or_objs]
        unwrap = True
        record_dependency_call(objs[0])
        objs[0].logger.debug(".create called for %s", objs[0])
    else:
        if not isinstance(obj_or_objs, Sequence):
            raise TypeError(
                "_load_or_create() expected a Spec object or a sequence of Spec objects"
            )
        objs = list(obj_or_objs)
        unwrap = False
        if any(not isinstance(obj, Spec) for obj in objs):
            raise TypeError("_load_or_create() expected Spec objects")
        for obj in objs:
            record_dependency_call(obj)
    return objs, unwrap


@overload
def create[T](obj: Spec[T], *, on: Sequence[WorkerBackend] | None = None) -> T: ...
@overload
def create[T](
    objs: Sequence[Spec[T]], *, on: Sequence[WorkerBackend] | None = None
) -> list[T]: ...
def create[T](
    obj_or_objs: Spec[T] | Sequence[Spec[T]],
    *,
    on: Sequence[WorkerBackend] | None = None,
) -> T | list[T]:
    if on is not None:
        from furu.execution.execution_coordinator import ExecutionCoordinator

        objs = [obj_or_objs] if isinstance(obj_or_objs, Spec) else list(obj_or_objs)
        ExecutionCoordinator.run(objs, worker_backends=tuple(on))
    return _load_or_create(obj_or_objs)


def load_existing[T](objs: Sequence[Spec[T]]) -> list[T]:
    if not isinstance(objs, Sequence):
        raise TypeError("load_existing() expected a sequence of Spec objects")
    objs = list(objs)
    if any(not isinstance(obj, Spec) for obj in objs):
        raise TypeError("load_existing() expected Spec objects")
    loaded: list[T] = []
    missing: list[Spec[T]] = []
    for obj in objs:
        record_dependency_call(obj)
        if (result_dir := result_dir_for_loading(obj)) is None:
            raise_if_stale(obj)
            missing.append(obj)
            continue
        loaded.append(
            cast(
                T,
                load_result_bundle(
                    result_dir,
                    data_dir=data_dir_in(obj._base_dir),
                    declared_type=declared_result_type(type(obj)),
                ),
            )
        )
    if missing:
        if _worker_execution_lease_id.get() is not None:
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
    return loaded


def _cached_to_build_msg(cached: list[Spec[Any]], to_build: list[Spec[Any]]) -> str:
    def fmt(objs: list[Spec[Any]]) -> str:
        if len(cached) + len(to_build) > 5:
            return str(len(objs))
        return ", ".join(o._log_label for o in objs)

    msg = f"cached {fmt(cached)}"
    return f"building {fmt(to_build)}, {msg}" if to_build else msg


def _load_or_create_worker[T](
    obj_or_objs: Spec[T] | Sequence[Spec[T]],
) -> T | list[T]:
    objs, unwrap = _normalize_load_or_create_input(obj_or_objs)

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
                        data_dir=data_dir_in(obj._base_dir),
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

    if unwrap:
        (result,) = loaded
        return result
    return loaded


def _load_or_create_local[T](
    obj_or_objs: Spec[T] | Sequence[Spec[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    objs, unwrap = _normalize_load_or_create_input(obj_or_objs)

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
                    data_dir=data_dir_in(obj._base_dir),
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
            if (cached_result_dir := result_dir_for_loading(obj)) is not None:
                late_hits += 1
                results_by_object_id[obj.object_id] = cast(
                    T,
                    load_result_bundle(
                        cached_result_dir,
                        data_dir=data_dir_in(obj._base_dir),
                        declared_type=declared_result_type(type(obj)),
                    ),
                )
            else:
                pending.append(obj)

        if late_hits:
            objs[0].logger.info(
                "%d became ready while waiting, %d to build", late_hits, len(pending)
            )

        direct_create_started = unwrap and bool(pending)
        create_started_at = time.monotonic()
        if direct_create_started:
            objs[0].logger.info("creating %s", objs[0]._log_label)

        grouped: dict[type[object], list[Spec[T]]] = {}
        for obj in pending:
            grouped.setdefault(type(obj), []).append(obj)

        for group in grouped.values():
            _create_and_store_group(
                group,
                has_lock=has_lock,
                results_by_object_id=results_by_object_id,
            )

    outputs = [results_by_object_id[obj.object_id] for obj in objs]

    if unwrap:
        (obj,) = objs
        (output,) = outputs
        if direct_create_started:
            obj.logger.info(
                "finished %s ok · %s",
                obj._log_label,
                format_duration(time.monotonic() - create_started_at),
            )
        return output
    return outputs


def _create_and_store_group[T](
    group: list[Spec[T]],
    *,
    has_lock: HasLock,
    results_by_object_id: dict[str, T],
) -> None:
    log_paths = tuple(run_log_path_in(obj._base_dir) for obj in group)

    metadata = [RunningMetadata.write_for(obj) for obj in group]

    with _scoped_log_files(log_paths):
        logger = group[0].logger
        logger.debug("create start")
        group_started_at = time.monotonic()
        try:
            match group[0]._furu_create_mode:
                case "batched":
                    logger.debug("running create_batched()")
                    with (
                        dependency_recorder() as recorder,
                        _allow_direct_create(type(group[0])),
                    ):
                        results = type(group[0]).create_batched(group)
                    observed = recorder.finalize()
                    logger.debug("create_batched() returned")
                    if not isinstance(results, list):
                        raise TypeError(
                            f"{type(group[0]).__name__}.create_batched() must return a list"
                        )
                    # TODO: Track dependency calls per object during batched execution.
                    # This currently assigns dependencies observed anywhere in the batch
                    # to every object.
                    observed_dependencies = [observed for _ in group]
                case "single":
                    logger.debug("running sequential create() fallback")
                    results = []
                    observed_dependencies = []
                    for obj in group:
                        with (
                            dependency_recorder() as recorder,
                            _allow_direct_create(obj),
                        ):
                            results.append(obj.create())
                        observed_dependencies.append(recorder.finalize())
                    logger.debug("sequential create() fallback returned")
                case None:
                    raise TypeError(
                        f"{type(group[0]).__qualname__} cannot create missing results because it does not "
                        "define create() or create_batched()"
                    )
                case _:
                    assert_never(group[0]._furu_create_mode)

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
                )

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
