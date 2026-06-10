from __future__ import annotations

import functools
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import (
    Any,
    TypeVar,
    assert_never,
    cast,
    get_args,
    get_origin,
    overload,
)

from furu._storage_layout import (
    compute_lock_path_in,
    metadata_path_in,
    result_dir_in,
    run_log_path_in,
)
from furu.core import Furu, FuruCreateMode
from furu.dependencies import dependency_recorder, record_dependency_call
from furu.locking import lock_many
from furu.logging import _scoped_log_files
from furu.metadata import RunningMetadata
from furu.migration import result_dir_for_loading
from furu.result import _save_result_bundle, load_result_bundle
from furu.result.save_as import _unwrap_save_as
from furu.utils import class_label, nfs_safe_unique_name
from furu.worker.context import (
    _DependencyNotReady,
    _worker_execution_lease_id,
)

type HasLock = Callable[[], bool]
type _DirectCreateTarget = Furu[Any] | type[Furu[Any]]


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


def _install_create_dispatchers[T](cls: type[Furu[T]]) -> None:
    if "create" in cls.__dict__:
        raw_create = cls.__dict__["create"]

        @functools.wraps(raw_create)
        def create_dispatcher(self: Furu[T], *args: Any, **kwargs: Any) -> T:
            if _direct_create_target.get() is self:
                return raw_create(self, *args, **kwargs)
            return _load_or_create(self, *args, **kwargs)

        setattr(cls, "create", create_dispatcher)

    if "create_batched" in cls.__dict__:
        raw_create_batched = cls.__dict__["create_batched"]
        func = raw_create_batched.__func__

        @functools.wraps(func)
        def create_batched_guard(
            owner: type[Furu[T]], *args: Any, **kwargs: Any
        ) -> list[T]:
            target = _direct_create_target.get()
            if not (isinstance(target, type) and issubclass(target, owner)):
                raise RuntimeError(
                    f"{owner.__name__}.create_batched() must not be called directly; "
                    "call .create() on Furu objects instead"
                )
            return func(owner, *args, **kwargs)

        setattr(cls, "create_batched", classmethod(create_batched_guard))


def _resolve_create_mode[T](cls: type[Furu[T]]) -> FuruCreateMode:
    defines_single = False
    defines_batched = False

    for base in cls.__mro__:
        if not issubclass(base, Furu) or base is Furu:
            continue

        if "create" in base.__dict__:
            defines_single = True
        if "create_batched" in base.__dict__:
            if not isinstance(base.__dict__["create_batched"], classmethod):
                raise TypeError(
                    f"{class_label(base)}.create_batched must be a @classmethod"
                )
            defines_batched = True

    if defines_single and defines_batched:
        raise TypeError(
            f"{class_label(cls)} must define exactly one of create or create_batched"
        )
    if defines_single:
        return "single"
    if defines_batched:
        return "batched"
    return None


def _store_result[T](
    obj: Furu[T],
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

    declared_type: object = Any
    for cls in type(obj).__mro__:
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Furu:
                declared_type = get_args(base)[0]
                break
        else:
            continue
        break

    if isinstance(declared_type, TypeVar) or any(
        isinstance(arg, TypeVar) for arg in get_args(declared_type)
    ):
        raise TypeError(
            f"{type(obj).__name__} must declare its concrete result type directly as Furu[...]"
        )

    should_load_after_dump = _save_result_bundle(
        result,
        tmp_result_dir,
        declared_type=declared_type,
        result_codecs=obj.result_codecs,
    )

    if not has_lock():
        raise RuntimeError(f"lost lock at {lock_path} after writing temporary result")

    tmp_result_dir.rename(result_dir)

    metadata_text = metadata.to_complete(
        observed_dependencies=observed_dependencies
    ).model_dump_json(indent=2)
    metadata_path_in(obj._base_dir).write_text(metadata_text)

    obj.logger.debug("stored result bundle at %s", result_dir)
    if should_load_after_dump:
        return cast(T, load_result_bundle(result_dir))
    return _unwrap_save_as(result)


def _format_error_debug_details(exc: BaseException) -> str:
    parts = ["Traceback (most recent call last):\n"]
    parts.extend(
        traceback.format_list(
            traceback.extract_stack()[:-2] + traceback.extract_tb(exc.__traceback__)
        )
    )
    parts.extend(traceback.format_exception_only(type(exc), exc))
    parts.append("\n=== Debug Traceback ===\n")
    parts.extend(
        traceback.TracebackException.from_exception(exc, capture_locals=False).format(
            chain=True
        )
    )
    return "".join(parts)


@overload
def _load_or_create[T](obj: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def _load_or_create[T](
    objs: Sequence[Furu[T]], *, use_lock: bool = True
) -> list[T]: ...


def _load_or_create[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    if _worker_execution_lease_id.get() is not None:
        return _load_or_create_worker(obj_or_objs)
    return _load_or_create_local(obj_or_objs, use_lock=use_lock)


def _ensure_single_result[T](obj: Furu[T]) -> None:
    if (cached_result_dir := result_dir_for_loading(obj)) is not None:
        obj.logger.info("cache hit for %s at %s", obj._log_label, cached_result_dir)
        return

    obj._base_dir.mkdir(parents=True, exist_ok=True)

    with lock_many([compute_lock_path_in(obj._base_dir)]) as has_lock:
        if (cached_result_dir := result_dir_for_loading(obj)) is not None:
            obj.logger.info(
                "cache hit for %s after waiting at %s",
                obj._log_label,
                cached_result_dir,
            )
            return

        _create_and_store_group(
            [obj],
            has_lock=has_lock,
            results_by_object_id={},
        )


def _normalize_load_or_create_input[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
) -> tuple[list[Furu[T]], bool]:
    if isinstance(obj_or_objs, Furu):
        objs = [obj_or_objs]
        unwrap = True
        record_dependency_call(objs[0])
        objs[0].logger.debug(".create called for %s", objs[0])
    else:
        if not isinstance(obj_or_objs, Sequence):
            raise TypeError(
                "_load_or_create() expected a Furu object or a sequence of Furu objects"
            )
        objs = list(obj_or_objs)
        unwrap = False
        if any(not isinstance(obj, Furu) for obj in objs):
            raise TypeError("_load_or_create() expected Furu objects")
        for obj in objs:
            record_dependency_call(obj)
    return objs, unwrap


def _load_or_create_worker[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
) -> T | list[T]:
    objs, unwrap = _normalize_load_or_create_input(obj_or_objs)

    loaded: list[T] = []
    missing: list[Furu[T]] = []

    for obj in objs:
        if (cached_result_dir := result_dir_for_loading(obj)) is not None:
            obj.logger.info("cache hit for %s at %s", obj._log_label, cached_result_dir)
            loaded.append(cast(T, load_result_bundle(cached_result_dir)))
        else:
            missing.append(obj)

    if missing:
        raise _DependencyNotReady(
            dependencies=missing,
            call_kind="create",
        )

    if unwrap:
        (obj,) = objs
        (result,) = loaded
        return result
    return loaded


def _load_or_create_local[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    objs, unwrap = _normalize_load_or_create_input(obj_or_objs)

    if not objs:
        return []

    unique_by_object_id: dict[str, Furu[T]] = {}
    for obj in objs:
        unique_by_object_id.setdefault(obj.object_id, obj)
    unique = list(unique_by_object_id.values())

    results_by_object_id: dict[str, T] = {}
    missing: list[Furu[T]] = []

    for obj in unique:
        if (cached_result_dir := result_dir_for_loading(obj)) is not None:
            obj.logger.info("cache hit for %s at %s", obj._log_label, cached_result_dir)
            results_by_object_id[obj.object_id] = cast(
                T, load_result_bundle(cached_result_dir)
            )
        else:
            obj._base_dir.mkdir(parents=True, exist_ok=True)
            missing.append(obj)

    lock_ctx = (
        lock_many([compute_lock_path_in(obj._base_dir) for obj in missing])
        if use_lock and missing
        else nullcontext()
    )

    with lock_ctx as maybe_has_lock:
        has_lock = maybe_has_lock or (lambda: True)
        pending: list[Furu[T]] = []
        for obj in missing:
            if (cached_result_dir := result_dir_for_loading(obj)) is not None:
                obj.logger.info(
                    "cache hit for %s after waiting at %s",
                    obj._log_label,
                    cached_result_dir,
                )
                results_by_object_id[obj.object_id] = cast(
                    T, load_result_bundle(cached_result_dir)
                )
            else:
                pending.append(obj)

        direct_create_started = unwrap and bool(pending)
        if direct_create_started:
            objs[0].logger.info("creating %s", obj._log_label)

        grouped: dict[type[object], list[Furu[T]]] = {}
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
            obj.logger.info("%s.create() finished", obj._log_label)
        return output
    return outputs


def _create_and_store_group[T](
    group: list[Furu[T]],
    *,
    has_lock: HasLock,
    results_by_object_id: dict[str, T],
) -> None:
    log_paths = tuple(run_log_path_in(obj._base_dir) for obj in group)

    metadata = [RunningMetadata.write_for(obj) for obj in group]

    with _scoped_log_files(log_paths):
        logger = group[0].logger
        logger.debug("create start")
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
                        f"{class_label(type(group[0]))} cannot create missing results because it does not "
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

            logger.debug("create complete")
        except _DependencyNotReady as exc:
            logger.debug(
                "create deferred: %s discovered %d missing dependency/dependencies",
                exc.call_kind,
                len(exc.dependencies),
            )
            raise
        except Exception as exc:
            logger.exception("create failed")
            logger.error("debug traceback:\n%s", _format_error_debug_details(exc))
            raise
