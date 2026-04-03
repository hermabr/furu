from __future__ import annotations

import os
import pickle
import secrets
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, overload

from furu.core import Furu
from furu.logging import _scoped_log_file, _scoped_log_files
from furu.locking import LockLostError, lock_many
from furu.metadata import RunningMetadata
from furu.utils import class_label

type _AnyFuru = Furu[Any]
type _HasLock = Callable[[], bool] | None
type FuruCreateMode = Literal["single", "batched"]

_CURRENT_EXECUTING_DATA_DIRS: ContextVar[frozenset[Path]] = ContextVar(
    "furu_current_executing_data_dirs",
    default=frozenset(),
)


def resolve_create_mode(cls: type[Furu[Any]]) -> FuruCreateMode:
    defines_single = "_create" in cls.__dict__
    defines_batched = "_create_batched" in cls.__dict__

    if defines_single and defines_batched:
        raise TypeError(
            f"{class_label(cls)} cannot define both _create and "
            "_create_batched in the same class body"
        )

    if defines_single:
        return "single"
    if defines_batched:
        if not isinstance(cls.__dict__["_create_batched"], classmethod):
            raise TypeError(
                f"{class_label(cls)}._create_batched must be declared as a "
                "@classmethod"
            )
        return "batched"

    for base in cls.__mro__[1:]:
        mode = base.__dict__.get("_furu_create_mode")
        if mode in ("single", "batched"):
            return mode

    raise TypeError(
        f"{class_label(cls)} must define either _create or "
        "_create_batched, or inherit one resolved mode"
    )


def _normalize_batch_input(
    obj_or_objs: _AnyFuru | Sequence[_AnyFuru],
) -> tuple[list[_AnyFuru], bool]:
    if isinstance(obj_or_objs, Furu):
        return [obj_or_objs], True

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
    return objs, False


def _canonical_data_dir(obj: _AnyFuru) -> Path:
    return obj.data_dir.resolve(strict=False)


def _dedupe_by_data_dir(objs: Sequence[_AnyFuru]) -> list[_AnyFuru]:
    deduped: dict[Path, _AnyFuru] = {}
    for obj in objs:
        deduped.setdefault(_canonical_data_dir(obj), obj)
    return list(deduped.values())


def _plan_execution(
    unique_objs: Sequence[_AnyFuru],
) -> tuple[list[_AnyFuru], list[_AnyFuru]]:
    existing: list[_AnyFuru] = []
    missing: list[_AnyFuru] = []
    for obj in unique_objs:
        if obj._result_path.exists():
            existing.append(obj)
        else:
            missing.append(obj)
    return existing, missing


def _group_pending_items(pending: Sequence[_AnyFuru]) -> list[list[_AnyFuru]]:
    grouped: dict[type[Any], list[_AnyFuru]] = {}
    for obj in pending:
        grouped.setdefault(type(obj), []).append(obj)
    return list(grouped.values())


def _load_result_from_disk(obj: Furu[Any]) -> Any:
    with obj._result_path.open("rb") as f:
        return pickle.load(f)


def _write_running_metadata(obj: _AnyFuru) -> RunningMetadata:
    metadata = RunningMetadata(
        artifact=obj.artifact,
        artifact_hash=obj.artifact_hash,
        schema_=obj.schema,
        schema_hash=obj.schema_hash,
        data_path=obj.data_dir.resolve(strict=False),
        started_at=datetime.now(timezone.utc),
    )
    obj._metadata_path.write_text(metadata.model_dump_json(indent=2))
    return metadata


def _store_result(
    obj: _AnyFuru,
    result: Any,
    *,
    metadata: RunningMetadata,
    has_lock: _HasLock,
) -> None:
    if has_lock is not None and not has_lock():
        raise LockLostError(f"lost lock at {obj._lock_path} before writing final result")

    tmp_result_path = obj._result_path.with_suffix(".pkl.tmp")
    with tmp_result_path.open("wb") as f:
        pickle.dump(result, f)
        f.flush()
        os.fsync(f.fileno())

    if has_lock is not None and not has_lock():
        raise LockLostError(
            f"lost lock at {obj._lock_path} after writing temporary result"
        )

    tmp_result_path.rename(obj._result_path)
    obj._metadata_path.write_text(metadata.to_complete().model_dump_json(indent=2))
    obj.logger.debug("stored result at %s", obj._result_path)


def _materialize_outputs(
    *,
    original_objs: Sequence[_AnyFuru],
    unique_objs: Sequence[_AnyFuru],
    newly_computed: dict[Path, Any],
) -> list[Any]:
    results_by_data_dir = dict(newly_computed)
    for obj in unique_objs:
        key = _canonical_data_dir(obj)
        results_by_data_dir.setdefault(key, _load_result_from_disk(obj))
    return [results_by_data_dir[_canonical_data_dir(obj)] for obj in original_objs]


def _write_error_logs(objs: Sequence[_AnyFuru], exc: BaseException) -> None:
    timestamp = datetime.now().strftime("%y%m%d_%H-%M-%S")
    suffix = secrets.token_hex(4)
    error_text = _format_error_log(exc)
    for obj in objs:
        obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
        error_path = obj._internal_furu_dir / f"error-{timestamp}-{suffix}.log"
        error_path.write_text(error_text, encoding="utf-8")


@overload
def load_or_create[T](obj: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def load_or_create[T](objs: Sequence[Furu[T]], *, use_lock: bool = True) -> list[T]: ...


@overload
def load_or_create(objs: Sequence[Furu[Any]], *, use_lock: bool = True) -> list[Any]: ...


def load_or_create(
    obj_or_objs: _AnyFuru | Sequence[_AnyFuru],
    *,
    use_lock: bool = True,
) -> Any | list[Any]:
    objs, unwrap = _normalize_batch_input(obj_or_objs)
    if not objs:
        return []

    _raise_if_reentrant(objs)

    if unwrap:
        obj = objs[0]
        obj.logger.info("calling %s.load_or_create()", obj._log_label)

    unique_objs = _dedupe_by_data_dir(objs)
    existing, missing = _plan_execution(unique_objs)
    for obj in existing:
        obj.logger.info("cache hit for %s at %s", obj._log_label, obj._result_path)

    for obj in missing:
        obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)

    newly_computed: dict[Path, Any] = {}
    lock_context = (
        lock_many([obj._lock_path for obj in missing])
        if use_lock and missing
        else nullcontext(None)
    )
    with lock_context as has_lock:
        pending = [obj for obj in missing if not obj._result_path.exists()]
        for obj in missing:
            if obj not in pending and obj._result_path.exists():
                obj.logger.info(
                    "cache hit for %s after waiting at %s",
                    obj._log_label,
                    obj._result_path,
                )

        for group in _group_pending_items(pending):
            newly_computed.update(_execute_group(group, has_lock=has_lock))

    outputs = _materialize_outputs(
        original_objs=objs,
        unique_objs=unique_objs,
        newly_computed=newly_computed,
    )

    if unwrap:
        obj = objs[0]
        obj.logger.info("%s.load_or_create() returned", obj._log_label)
        return outputs[0]
    return outputs


def _execute_group(group: list[_AnyFuru], *, has_lock: _HasLock) -> dict[Path, Any]:
    mode = group[0]._furu_create_mode
    if mode == "batched":
        metadata_by_data_dir = {
            _canonical_data_dir(obj): _write_running_metadata(obj) for obj in group
        }
        return _execute_batched_group(
            group,
            metadata_by_data_dir=metadata_by_data_dir,
            has_lock=has_lock,
        )
    if mode == "single":
        return _execute_single_group(group, has_lock=has_lock)
    raise AssertionError(f"unexpected Furu create mode: {mode}")


def _execute_batched_group(
    group: list[_AnyFuru],
    *,
    metadata_by_data_dir: dict[Path, RunningMetadata],
    has_lock: _HasLock,
) -> dict[Path, Any]:
    log_paths = tuple(obj._log_path for obj in group)
    with _scoped_log_files(log_paths):
        logger = group[0].logger
        logger.debug("load_or_create start")
        try:
            with _reentry_guard(group):
                logger.debug("running _create_batched()")
                results = type(group[0])._create_batched(group)
                logger.debug("_create_batched() returned")

            if not isinstance(results, list):
                raise TypeError(
                    f"{type(group[0]).__name__}._create_batched() must return a list"
                )
            if len(results) != len(group):
                raise TypeError(
                    f"{type(group[0]).__name__}._create_batched() returned "
                    f"{len(results)} results for {len(group)} objects"
                )

            stored: dict[Path, Any] = {}
            for obj, result in zip(group, results, strict=True):
                _store_result(
                    obj,
                    result,
                    metadata=metadata_by_data_dir[_canonical_data_dir(obj)],
                    has_lock=has_lock,
                )
                stored[_canonical_data_dir(obj)] = result
            logger.debug("load_or_create complete")
            return stored
        except BaseException as exc:
            logger.exception("load_or_create failed")
            _write_error_logs(group, exc)
            raise


def _execute_single_group(
    group: list[_AnyFuru],
    *,
    has_lock: _HasLock,
) -> dict[Path, Any]:
    stored: dict[Path, Any] = {}
    for obj in group:
        with _scoped_log_file(obj._log_path):
            logger = obj.logger
            logger.debug("load_or_create start")
            try:
                with _reentry_guard([obj]):
                    metadata = _write_running_metadata(obj)
                    logger.debug("running _create()")
                    result = obj._create()
                    logger.debug("_create() returned")
                _store_result(
                    obj,
                    result,
                    metadata=metadata,
                    has_lock=has_lock,
                )
                logger.debug("load_or_create complete")
            except BaseException as exc:
                logger.exception("load_or_create failed")
                _write_error_logs([obj], exc)
                raise
        stored[_canonical_data_dir(obj)] = result
    return stored


@contextmanager
def _reentry_guard(objs: Sequence[_AnyFuru]):
    keys = _reentry_keys(objs)
    _raise_if_reentrant(objs)
    active_keys = _CURRENT_EXECUTING_DATA_DIRS.get()
    token = _CURRENT_EXECUTING_DATA_DIRS.set(active_keys | keys)
    try:
        yield
    finally:
        _CURRENT_EXECUTING_DATA_DIRS.reset(token)


def _raise_if_reentrant(objs: Sequence[_AnyFuru]) -> None:
    overlapping = _CURRENT_EXECUTING_DATA_DIRS.get() & _reentry_keys(objs)
    if not overlapping:
        return

    overlapping_paths = ", ".join(sorted(str(path) for path in overlapping))
    raise RuntimeError(
        "load_or_create() re-entered for objects already being created: "
        f"{overlapping_paths}"
    )


def _reentry_keys(objs: Sequence[_AnyFuru]) -> frozenset[Path]:
    return frozenset(_canonical_data_dir(obj) for obj in objs)


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
