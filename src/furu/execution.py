from __future__ import annotations

import secrets
import traceback
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import overload

from furu.core import Furu
from furu.locking import lock_many
from furu.logging import _scoped_log_files

_CURRENT_EXECUTING_CACHE_KEYS: ContextVar[frozenset[Path]] = ContextVar(
    "furu_current_executing_cache_keys",
    default=frozenset(),
)


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


def _assert_homogeneous_concrete_class[T](
    objs: Sequence[Furu[T]],
) -> type[Furu[T]]:
    cls = type(objs[0])
    if any(type(obj) is not cls for obj in objs[1:]):
        raise TypeError(
            "load_or_create() requires every batch entry to have the same concrete "
            "Furu class"
        )
    return cls


def _check_not_reentrant[T](objs: Sequence[Furu[T]]) -> None:
    overlapping = _CURRENT_EXECUTING_CACHE_KEYS.get() & frozenset(
        obj.cache_key for obj in objs
    )
    if not overlapping:
        return

    overlapping_paths = ", ".join(sorted(str(path) for path in overlapping))
    raise RuntimeError(
        "load_or_create() re-entered for objects already being created: "
        f"{overlapping_paths}"
    )


@contextmanager
def _execution_scope[T](objs: Sequence[Furu[T]]):
    active_keys = _CURRENT_EXECUTING_CACHE_KEYS.get()
    token = _CURRENT_EXECUTING_CACHE_KEYS.set(
        active_keys | frozenset(obj.cache_key for obj in objs)
    )
    try:
        yield
    finally:
        _CURRENT_EXECUTING_CACHE_KEYS.reset(token)


def _write_error_logs[T](objs: Sequence[Furu[T]], exc: BaseException) -> None:
    timestamp = datetime.now().strftime("%y%m%d_%H-%M-%S")
    suffix = secrets.token_hex(4)
    error_text = _format_error_log(exc)
    for obj in objs:
        obj.ensure_private_dir()
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

    _check_not_reentrant(objs)
    cls = _assert_homogeneous_concrete_class(objs)

    if unwrap:
        obj = objs[0]
        obj.logger.info("calling %s.load_or_create()", obj._log_label)

    unique_by_key: dict[Path, Furu[T]] = {}
    for obj in objs:
        unique_by_key.setdefault(obj.cache_key, obj)
    unique = list(unique_by_key.values())

    missing: list[Furu[T]] = []
    for obj in unique:
        if obj._result_path.exists():
            obj.logger.info("cache hit for %s at %s", obj._log_label, obj._result_path)
        else:
            obj.ensure_private_dir()
            missing.append(obj)

    computed: dict[Path, T] = {}
    ctx = (
        lock_many([obj._lock_path for obj in missing])
        if use_lock and missing
        else nullcontext(None)
    )

    with ctx as lease:
        pending: list[Furu[T]] = []
        for obj in missing:
            if obj._result_path.exists():
                obj.logger.info(
                    "cache hit for %s after waiting at %s",
                    obj._log_label,
                    obj._result_path,
                )
            else:
                pending.append(obj)

        if pending:
            uses_custom_many = (
                cls._create_many.__func__ is not Furu._create_many.__func__
            )

            if uses_custom_many:
                with (
                    _execution_scope(pending),
                    _scoped_log_files(tuple(obj._log_path for obj in pending)),
                ):
                    logger = pending[0].logger
                    logger.debug("load_or_create start")
                    try:
                        metadata_by_key = {
                            obj.cache_key: obj.write_running_metadata()
                            for obj in pending
                        }
                        logger.debug("running _create_many()")
                        results = cls._create_many(pending)
                        logger.debug("_create_many() returned")
                        if not isinstance(results, list):
                            raise TypeError(
                                f"{cls.__name__}._create_many() must return a list"
                            )
                        if len(results) != len(pending):
                            raise TypeError(
                                f"{cls.__name__}._create_many() returned "
                                f"{len(results)} results for {len(pending)} objects"
                            )

                        for obj, result in zip(pending, results, strict=True):
                            obj.commit(
                                result,
                                metadata=metadata_by_key[obj.cache_key],
                                lease=lease,
                            )
                            computed[obj.cache_key] = result
                        logger.debug("load_or_create complete")
                    except BaseException as exc:
                        logger.exception("load_or_create failed")
                        _write_error_logs(pending, exc)
                        raise
            else:
                for obj in pending:
                    with _execution_scope([obj]), _scoped_log_files((obj._log_path,)):
                        logger = obj.logger
                        logger.debug("load_or_create start")
                        try:
                            metadata = obj.write_running_metadata()
                            logger.debug("running _create()")
                            result = obj._create()
                            logger.debug("_create() returned")
                            obj.commit(
                                result,
                                metadata=metadata,
                                lease=lease,
                            )
                            computed[obj.cache_key] = result
                            logger.debug("load_or_create complete")
                        except BaseException as exc:
                            logger.exception("load_or_create failed")
                            _write_error_logs([obj], exc)
                            raise

    results_by_key = dict(computed)
    for obj in unique:
        results_by_key.setdefault(obj.cache_key, obj.load_result())

    outputs = [results_by_key[obj.cache_key] for obj in objs]
    if unwrap:
        obj = objs[0]
        obj.logger.info("%s.load_or_create() returned", obj._log_label)
        return outputs[0]
    return outputs
