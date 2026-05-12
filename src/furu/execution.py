from __future__ import annotations

import functools
import traceback
from collections import deque
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

from furu.core import Furu, FuruCreateMode
from furu.dag import (
    FuruDagNode,
    add_dependency,
    add_execution_dag_nodes,
    make_execution_dag,
)
from furu.dependencies import dependency_recorder, record_dependency_call
from furu.locking import LockLostError, lock_many
from furu.logging import _scoped_log_files
from furu.metadata import RunningMetadata
from furu.migration import result_dir_for_loading
from furu.result import load_result_bundle, save_result_bundle
from furu.result.save_as import _unwrap_save_as
from furu.storage_layout import (
    compute_lock_path_in,
    internal_furu_dir_in,
    metadata_path_in,
    result_dir_in,
    run_log_path_in,
)
from furu.utils import class_label, nfs_safe_unique_name
from furu.worker_execution import (
    _DependencyNotReady,
    _worker_execution_lease_id,
    worker_execution_context,
)

type HasLock = Callable[[], bool]
type _AnyFuruNode = FuruDagNode[Furu[Any]]


_create_execution_active: ContextVar[bool] = ContextVar(
    "_furu_create_execution_active", default=False
)


@contextmanager
def _allow_direct_create() -> Iterator[None]:
    token = _create_execution_active.set(True)
    try:
        yield
    finally:
        _create_execution_active.reset(token)


def _install_create_guards(cls: type[Furu[Any]]) -> None:
    for attr in ("create", "create_batched"):
        if attr not in cls.__dict__:
            continue
        is_batched = attr == "create_batched"
        raw = cls.__dict__[attr]
        func = raw.__func__ if is_batched else raw

        @functools.wraps(func)
        def guarded(first: Any, *args: Any, **kwargs: Any) -> Any:
            if not _create_execution_active.get():
                owner = first.__name__ if is_batched else type(first).__name__
                suggestion = (
                    "furu.load_or_create()" if is_batched else ".load_or_create()"
                )
                raise RuntimeError(
                    f"{owner}.{attr}() must not be called directly; "
                    f"call {suggestion} instead"
                )
            return func(first, *args, **kwargs)

        setattr(cls, attr, classmethod(guarded) if is_batched else guarded)


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
    raise TypeError(
        f"{class_label(cls)} must define exactly one create hook in its inheritance chain"
    )


def _store_result[T](
    obj: Furu[T],
    result: T,
    *,
    metadata: RunningMetadata,
    observed_dependencies: tuple[str, ...],
    has_lock: HasLock,
) -> None:
    lock_path = compute_lock_path_in(obj.data_dir)
    result_dir = result_dir_in(obj.data_dir)
    if not has_lock():
        raise LockLostError(f"lost lock at {lock_path} before writing final result")

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

    save_result_bundle(
        result,
        tmp_result_dir,
        declared_type=declared_type,
        registry=obj.result_registry,
    )

    if not has_lock():
        raise LockLostError(f"lost lock at {lock_path} after writing temporary result")

    tmp_result_dir.rename(result_dir)

    metadata_text = metadata.to_complete(
        observed_dependencies=observed_dependencies
    ).model_dump_json(indent=2)
    metadata_path_in(obj.data_dir).write_text(metadata_text)

    obj.logger.debug("stored result bundle at %s", result_dir)


def _format_error_debug_details(exc: BaseException) -> str:
    parts = ["Traceback (most recent call last):\n"]
    parts.extend(
        traceback.format_list(
            traceback.extract_stack()[:-2] + traceback.extract_tb(exc.__traceback__)
        )
    )
    parts.extend(traceback.format_exception_only(type(exc), exc))
    parts.append("\n=== Debug Details (with locals) ===\n")
    parts.extend(
        traceback.TracebackException.from_exception(exc, capture_locals=True).format(
            chain=True
        )
    )
    return "".join(parts)


@overload
def load_or_create[T](obj: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def load_or_create[T](objs: Sequence[Furu[T]], *, use_lock: bool = True) -> list[T]: ...


def load_or_create[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    if _worker_execution_lease_id.get() is not None:
        return _load_or_create_worker(obj_or_objs)
    return _load_or_create_local(obj_or_objs, use_lock=use_lock)


def _enqueue_zero_dependency_node(
    node: _AnyFuruNode,
    zero_dependency_nodes: deque[_AnyFuruNode],
    queued_node_ids: set[str],
) -> None:
    node_id = node.obj.object_id
    if node.dependencies or node_id in queued_node_ids:
        return
    zero_dependency_nodes.append(node)
    queued_node_ids.add(node_id)


def _add_lazy_dependencies_to_execution_dag(
    *,
    node: _AnyFuruNode,
    dependencies: Sequence[Furu[Any]],
    nodes_by_id: dict[str, _AnyFuruNode],
    zero_dependency_nodes: deque[_AnyFuruNode],
    queued_node_ids: set[str],
) -> None:
    added_nodes = add_execution_dag_nodes(dependencies, nodes_by_id)

    for dependency in dependencies:
        dependency_node = nodes_by_id[dependency.object_id]
        add_dependency(node, dependency_node)
        _enqueue_zero_dependency_node(
            dependency_node,
            zero_dependency_nodes,
            queued_node_ids,
        )

    for added_node in added_nodes:
        _enqueue_zero_dependency_node(
            added_node,
            zero_dependency_nodes,
            queued_node_ids,
        )


def _complete_execution_dag_node(
    node: _AnyFuruNode,
    *,
    nodes_by_id: dict[str, _AnyFuruNode],
    zero_dependency_nodes: deque[_AnyFuruNode],
    queued_node_ids: set[str],
) -> None:
    node_id = node.obj.object_id
    if nodes_by_id.pop(node_id, None) is None:
        return

    for dependency in node.dependencies:
        dependency.dependents = [
            dependent
            for dependent in dependency.dependents
            if dependent.obj.object_id != node_id
        ]

    for dependent in node.dependents:
        dependent.dependencies = [
            dependency
            for dependency in dependent.dependencies
            if dependency.obj.object_id != node_id
        ]
        if dependent.obj.object_id in nodes_by_id:
            _enqueue_zero_dependency_node(
                dependent,
                zero_dependency_nodes,
                queued_node_ids,
            )

    node.dependencies.clear()
    node.dependents.clear()


def _format_unresolved_dag_nodes(nodes_by_id: dict[str, _AnyFuruNode]) -> str:
    labels = [
        f"{type(node.obj).__name__}:{node.obj.artifact_hash[:5]}"
        for node in nodes_by_id.values()
    ]
    if len(labels) <= 5:
        return ", ".join(labels)
    return ", ".join(labels[:5]) + f", and {len(labels) - 5} more"


def submit(objs: Sequence[Furu[Any]]) -> None:
    zero_dependency_nodes_list, nodes_by_id = make_execution_dag(objs)
    zero_dependency_nodes = deque(zero_dependency_nodes_list)
    queued_node_ids = {node.obj.object_id for node in zero_dependency_nodes}

    while zero_dependency_nodes:
        node = zero_dependency_nodes.popleft()
        node_id = node.obj.object_id
        queued_node_ids.discard(node_id)
        if nodes_by_id.get(node_id) is not node or node.dependencies:
            continue

        try:
            with worker_execution_context(lease_id=node_id):
                _load_or_create_local(node.obj)
        except _DependencyNotReady as exc:
            _add_lazy_dependencies_to_execution_dag(
                node=node,
                dependencies=exc.dependencies,
                nodes_by_id=nodes_by_id,
                zero_dependency_nodes=zero_dependency_nodes,
                queued_node_ids=queued_node_ids,
            )
            continue

        _complete_execution_dag_node(
            node,
            nodes_by_id=nodes_by_id,
            zero_dependency_nodes=zero_dependency_nodes,
            queued_node_ids=queued_node_ids,
        )

    if nodes_by_id:
        raise RuntimeError(
            "submit() could not make progress through the execution DAG; "
            f"unresolved nodes: {_format_unresolved_dag_nodes(nodes_by_id)}"
        )


def _normalize_load_or_create_input[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
) -> tuple[list[Furu[T]], bool]:
    if isinstance(obj_or_objs, Furu):
        objs = [obj_or_objs]
        unwrap = True
        record_dependency_call(objs[0])
        objs[0].logger.info("calling %s.load_or_create()", objs[0]._log_label)
    else:
        if not isinstance(obj_or_objs, Sequence):
            raise TypeError(
                "load_or_create() expected a Furu object or a sequence of Furu objects"
            )
        objs = list(obj_or_objs)
        unwrap = False
        if any(not isinstance(obj, Furu) for obj in objs):
            raise TypeError("load_or_create() expected Furu objects")
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
            call_kind="load_or_create",
        )

    if unwrap:
        (obj,) = objs
        (result,) = loaded
        obj.logger.info("%s.load_or_create() returned", obj._log_label)
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
            internal_furu_dir_in(obj.data_dir).mkdir(parents=True, exist_ok=True)
            missing.append(obj)

    lock_ctx = (
        lock_many([compute_lock_path_in(obj.data_dir) for obj in missing])
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

        grouped: dict[type[object], list[Furu[T]]] = {}
        for obj in pending:
            grouped.setdefault(type(obj), []).append(obj)

        for group in grouped.values():
            _execute_group(
                group,
                has_lock=has_lock,
                results_by_object_id=results_by_object_id,
            )

    outputs = [results_by_object_id[obj.object_id] for obj in objs]

    if unwrap:
        (obj,) = objs
        (output,) = outputs
        obj.logger.info("%s.load_or_create() returned", obj._log_label)
        return output
    return outputs


def _execute_group[T](
    group: list[Furu[T]],
    *,
    has_lock: HasLock,
    results_by_object_id: dict[str, T],
) -> None:
    log_paths = tuple(run_log_path_in(obj.data_dir) for obj in group)

    metadata = [RunningMetadata.write_for(obj) for obj in group]

    with _scoped_log_files(log_paths), _allow_direct_create():
        logger = group[0].logger
        logger.debug("load_or_create start")
        try:
            match group[0]._furu_create_mode:
                case "batched":
                    logger.debug("running create_batched()")
                    with dependency_recorder() as recorder:
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
                        with dependency_recorder() as recorder:
                            results.append(obj.create())
                        observed_dependencies.append(recorder.finalize())
                    logger.debug("sequential create() fallback returned")
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
                _store_result(
                    obj,
                    result,
                    metadata=obj_metadata,
                    observed_dependencies=observed_dependency_ids,
                    has_lock=has_lock,
                )
                results_by_object_id[obj.object_id] = _unwrap_save_as(result)

            logger.debug("load_or_create complete")
        except _DependencyNotReady:
            raise
        except BaseException as exc:
            logger.exception("load_or_create failed")
            logger.error(
                "debug traceback with locals:\n%s", _format_error_debug_details(exc)
            )
            raise
