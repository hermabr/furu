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
from typing import assert_never, overload

from furu.core import Furu
from furu.locking import LockLostError, lock_many
from furu.logging import _scoped_log_files
from furu.metadata import RunningMetadata

_CURRENT_EXECUTING_KEYS: ContextVar[frozenset[Path]] = ContextVar(
    "furu_current_executing_keys",
    default=frozenset(),
)


def _always_has_lock() -> bool:
    return True


def _normalize_input[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
) -> tuple[list[Furu[T]], bool]:
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


def _dedupe[T](objs: Sequence[Furu[T]]) -> list[Furu[T]]:
    deduped: dict[Path, Furu[T]] = {}
    for obj in objs:
        deduped.setdefault(obj.cache_key, obj)
    return list(deduped.values())


def _iter_work_units[T](pending: Sequence[Furu[T]]) -> Iterator[list[Furu[T]]]:
    by_cls: dict[type[object], list[Furu[T]]] = {}
    for obj in pending:
        by_cls.setdefault(type(obj), []).append(obj)

    for group in by_cls.values():
        if group[0]._furu_create_mode == "batched":
            yield group
        else:
            yield from ([obj] for obj in group)


def _load_result[T](obj: Furu[T]) -> T:
    with obj._result_path.open("rb") as f:
        return pickle.load(f)


def _write_running_metadata[T](obj: Furu[T]) -> RunningMetadata:
    metadata = RunningMetadata(
        artifact=obj.artifact,
        artifact_hash=obj.artifact_hash,
        schema_=obj.schema,
        schema_hash=obj.schema_hash,
        data_path=obj.cache_key,
        started_at=datetime.now(timezone.utc),
    )
    obj._metadata_path.write_text(metadata.model_dump_json(indent=2))
    return metadata


def _store_result[T](
    obj: Furu[T],
    result: T,
    *,
    metadata: RunningMetadata,
    has_lock: Callable[[], bool],
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


def _run_work_unit[T](
    unit: list[Furu[T]],
    *,
    has_lock: Callable[[], bool],
) -> dict[Path, T]:
    if not unit:
        return {}

    cls = type(unit[0])
    log_paths = tuple(obj._log_path for obj in unit)
    metadata_by_key = {obj.cache_key: _write_running_metadata(obj) for obj in unit}

    with _scoped_log_files(log_paths):
        logger = unit[0].logger
        logger.debug("load_or_create start")
        try:
            mode = cls._furu_create_mode
            match mode:
                case "batched":
                    logger.debug("running _create_batched()")
                    results = cls._create_batched(unit)
                    logger.debug("_create_batched() returned")
                    if not isinstance(results, list):
                        raise TypeError(
                            f"{cls.__name__}._create_batched() must return a list"
                        )
                    if len(results) != len(unit):
                        raise TypeError(
                            f"{cls.__name__}._create_batched() returned "
                            f"{len(results)} results for {len(unit)} objects"
                        )
                case "single":
                    results = []
                    for obj in unit:
                        logger.debug("running _create()")
                        result = obj._create()
                        logger.debug("_create() returned")
                        results.append(result)
                case _:
                    assert_never(mode)

            stored: dict[Path, T] = {}
            for obj, result in zip(unit, results, strict=True):
                _store_result(
                    obj,
                    result,
                    metadata=metadata_by_key[obj.cache_key],
                    has_lock=has_lock,
                )
                stored[obj.cache_key] = result

            logger.debug("load_or_create complete")
            return stored
        except BaseException as exc:
            logger.exception("load_or_create failed")
            _write_error_logs(unit, exc)
            raise


@contextmanager
def _executing[T](objs: Sequence[Furu[T]]):
    keys = frozenset(obj.cache_key for obj in objs)
    active = _CURRENT_EXECUTING_KEYS.get()
    overlap = active & keys
    if overlap:
        raise RuntimeError(
            "load_or_create() re-entered for objects already being created: "
            + ", ".join(sorted(str(path) for path in overlap))
        )

    token = _CURRENT_EXECUTING_KEYS.set(active | keys)
    try:
        yield
    finally:
        _CURRENT_EXECUTING_KEYS.reset(token)


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


@overload
def load_or_create[T](obj: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def load_or_create[T](objs: Sequence[Furu[T]], *, use_lock: bool = True) -> list[T]: ...


def load_or_create[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    objs, unwrap = _normalize_input(obj_or_objs)
    if not objs:
        return []

    if unwrap:
        obj = objs[0]
        obj.logger.info("calling %s.load_or_create()", obj._log_label)

    unique = _dedupe(objs)
    results_by_key: dict[Path, T] = {}
    missing: list[Furu[T]] = []

    for obj in unique:
        if obj._result_path.exists():
            results_by_key[obj.cache_key] = _load_result(obj)
            obj.logger.info("cache hit for %s at %s", obj._log_label, obj._result_path)
        else:
            obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
            missing.append(obj)

    overlap = _CURRENT_EXECUTING_KEYS.get() & frozenset(
        obj.cache_key for obj in missing
    )
    if overlap:
        raise RuntimeError(
            "load_or_create() re-entered for objects already being created: "
            + ", ".join(sorted(str(path) for path in overlap))
        )

    lock_ctx = (
        lock_many([obj._lock_path for obj in missing])
        if use_lock and missing
        else nullcontext(_always_has_lock)
    )
    with lock_ctx as has_lock:
        pending: list[Furu[T]] = []
        for obj in missing:
            if obj._result_path.exists():
                results_by_key[obj.cache_key] = _load_result(obj)
                obj.logger.info(
                    "cache hit for %s after waiting at %s",
                    obj._log_label,
                    obj._result_path,
                )
            else:
                pending.append(obj)

        with _executing(pending):
            for unit in _iter_work_units(pending):
                results_by_key.update(_run_work_unit(unit, has_lock=has_lock))

    outputs = [results_by_key[obj.cache_key] for obj in objs]
    if unwrap:
        obj = objs[0]
        obj.logger.info("%s.load_or_create() returned", obj._log_label)
        return outputs[0]
    return outputs
