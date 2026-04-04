from __future__ import annotations

import os
import pickle
import secrets
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast, overload

from furu.core import Furu
from furu.locking import LockLostError, lock_many
from furu.logging import _scoped_log_files
from furu.metadata import RunningMetadata
from furu.utils import class_label

type HasLock = Callable[[], bool]

_CURRENT_EXECUTING_DATA_DIRS: ContextVar[frozenset[Path]] = ContextVar(
    "furu_current_executing_data_dirs",
    default=frozenset(),
)


def bind_create_many[T](cls: type[Furu[T]]) -> Callable[[list[Furu[T]]], list[T]]:
    has_single = "_create" in cls.__dict__
    has_batch = "_create_batched" in cls.__dict__

    if has_single and has_batch:
        raise TypeError(
            f"{class_label(cls)} cannot define both _create and "
            "_create_batched in the same class body"
        )

    if has_single:

        def create_many(objs: list[Furu[T]]) -> list[T]:
            return [obj._create() for obj in objs]

        return create_many

    if has_batch:
        declared_create_batched = cls.__dict__["_create_batched"]
        if not isinstance(declared_create_batched, classmethod):
            raise TypeError(
                f"{class_label(cls)}._create_batched must be declared as a @classmethod"
            )

        create_batched = declared_create_batched.__get__(None, cls)

        def create_many(objs: list[Furu[T]]) -> list[T]:
            return create_batched(objs)

        return create_many

    raise TypeError(
        f"{class_label(cls)} must define either _create or "
        "_create_batched in the same class body"
    )


def _always_true() -> bool:
    return True


def _load_result_from_disk[T](obj: Furu[T]) -> T:
    with obj._result_path.open("rb") as f:
        return pickle.load(f)


def _write_running_metadata[T](obj: Furu[T]) -> RunningMetadata:
    metadata = RunningMetadata(
        artifact=obj.artifact,
        artifact_hash=obj.artifact_hash,
        schema_=obj.schema,
        schema_hash=obj.schema_hash,
        data_path=obj._data_key,
        started_at=datetime.now(timezone.utc),
    )
    obj._metadata_path.write_text(metadata.model_dump_json(indent=2))
    return metadata


def _store_result[T](
    obj: Furu[T],
    result: T,
    *,
    metadata: RunningMetadata,
    has_lock: HasLock,
) -> None:
    if not has_lock():
        raise LockLostError(
            f"lost lock at {obj._lock_path} before writing final result"
        )

    tmp_result_path = obj._result_path.with_suffix(".pkl.tmp")
    with tmp_result_path.open("wb") as f:
        pickle.dump(result, f)
        f.flush()
        os.fsync(f.fileno())

    if not has_lock():
        raise LockLostError(
            f"lost lock at {obj._lock_path} after writing temporary result"
        )

    tmp_result_path.rename(obj._result_path)
    obj._metadata_path.write_text(metadata.to_complete().model_dump_json(indent=2))
    obj.logger.debug("stored result at %s", obj._result_path)


def _execute_class_group[T](
    group: list[Furu[T]],
    *,
    has_lock: HasLock,
) -> dict[Path, T]:
    cls = type(group[0])
    metadata_by_key = {obj._data_key: _write_running_metadata(obj) for obj in group}
    log_paths = tuple(obj._log_path for obj in group)

    with _scoped_log_files(log_paths):
        logger = group[0].logger
        logger.debug("load_or_create start")
        try:
            results = cls._furu_create_many(group)
            if not isinstance(results, list):
                raise TypeError(
                    f"{cls.__name__}._furu_create_many() must return a list"
                )
            if len(results) != len(group):
                raise TypeError(
                    f"{cls.__name__}._furu_create_many() returned "
                    f"{len(results)} results for {len(group)} objects"
                )

            stored: dict[Path, T] = {}
            for obj, result in zip(group, results, strict=True):
                _store_result(
                    obj,
                    result,
                    metadata=metadata_by_key[obj._data_key],
                    has_lock=has_lock,
                )
                stored[obj._data_key] = result

            logger.debug("load_or_create complete")
            return stored
        except BaseException as exc:
            logger.exception("load_or_create failed")
            _write_error_logs(group, exc)
            raise


@overload
def load_or_create[T](obj: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def load_or_create[T](objs: Sequence[Furu[T]], *, use_lock: bool = True) -> list[T]: ...


def load_or_create[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    unwrap = isinstance(obj_or_objs, Furu)
    if unwrap:
        objs = [cast(Furu[T], obj_or_objs)]
    else:
        if not isinstance(obj_or_objs, Sequence):
            raise TypeError(
                "load_or_create() expected a Furu object or a sequence of Furu objects"
            )
        objs = list(obj_or_objs)
        for obj in objs:
            if not isinstance(obj, Furu):
                raise TypeError(
                    "load_or_create() expected every batch entry to be a Furu object"
                )

    if not objs:
        return []

    if unwrap:
        obj = objs[0]
        obj.logger.info("calling %s.load_or_create()", obj._log_label)

    unique: list[Furu[Any]] = []
    seen: set[Path] = set()
    for obj in objs:
        if obj._data_key not in seen:
            seen.add(obj._data_key)
            unique.append(obj)

    missing: list[Furu[Any]] = []
    results_by_key: dict[Path, Any] = {}
    for obj in unique:
        if obj._result_path.exists():
            obj.logger.info("cache hit for %s at %s", obj._log_label, obj._result_path)
        else:
            obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
            missing.append(obj)

    lock_ctx = (
        lock_many([obj._lock_path for obj in missing])
        if use_lock and missing
        else nullcontext(_always_true)
    )

    with _guard_reentry(missing):
        with lock_ctx as has_lock:
            pending: list[Furu[Any]] = []
            for obj in missing:
                if obj._result_path.exists():
                    obj.logger.info(
                        "cache hit for %s after waiting at %s",
                        obj._log_label,
                        obj._result_path,
                    )
                else:
                    pending.append(obj)

            groups: dict[type[object], list[Furu[Any]]] = {}
            for obj in pending:
                groups.setdefault(type(obj), []).append(obj)

            for group in groups.values():
                results_by_key.update(_execute_class_group(group, has_lock=has_lock))

    for obj in unique:
        results_by_key.setdefault(obj._data_key, _load_result_from_disk(obj))

    outputs = [results_by_key[obj._data_key] for obj in objs]
    if unwrap:
        obj = objs[0]
        obj.logger.info("%s.load_or_create() returned", obj._log_label)
        return outputs[0]
    return outputs


@contextmanager
def _guard_reentry(objs: Sequence[Furu[Any]]) -> Iterator[None]:
    keys = frozenset(obj._data_key for obj in objs)
    active_keys = _CURRENT_EXECUTING_DATA_DIRS.get()
    overlapping = active_keys & keys
    if overlapping:
        overlapping_paths = ", ".join(sorted(str(path) for path in overlapping))
        raise RuntimeError(
            "load_or_create() re-entered for objects already being created: "
            f"{overlapping_paths}"
        )

    token = _CURRENT_EXECUTING_DATA_DIRS.set(active_keys | keys)
    try:
        yield
    finally:
        _CURRENT_EXECUTING_DATA_DIRS.reset(token)


def _write_error_logs[T](objs: Sequence[Furu[T]], exc: BaseException) -> None:
    timestamp = datetime.now().strftime("%y%m%d_%H-%M-%S")
    suffix = secrets.token_hex(4)
    error_text = _format_error_log(exc)
    for obj in objs:
        obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
        error_path = obj._internal_furu_dir / f"error-{timestamp}-{suffix}.log"
        error_path.write_text(error_text, encoding="utf-8")


def _format_error_log(exc: BaseException) -> str:
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
