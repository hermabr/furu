from __future__ import annotations

import secrets
import shutil
import traceback
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import assert_never, cast, overload

from furu.core import Furu, FuruCreateMode
from furu.locking import LockLostError, lock_many
from furu.logging import _scoped_log_files
from furu.metadata import RunningMetadata
from furu.result import load_result_bundle, save_result_bundle
from furu.utils import _nfs_safe_unique_name, class_label

type HasLock = Callable[[], bool]


def _resolve_create_mode[T](cls: type[Furu[T]]) -> FuruCreateMode:
    defines_single = False
    defines_batched = False

    for base in cls.__mro__:
        if not issubclass(base, Furu) or base is Furu:
            continue

        if "_create" in base.__dict__:
            defines_single = True
        if "_create_batched" in base.__dict__:
            if not isinstance(base.__dict__["_create_batched"], classmethod):
                raise TypeError(
                    f"{class_label(base)}._create_batched must be a @classmethod"
                )
            defines_batched = True

    if defines_single and defines_batched:
        raise TypeError(
            f"{class_label(cls)} must define exactly one of _create or _create_batched"
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
    has_lock: HasLock,
) -> T:
    if not has_lock():
        raise LockLostError(
            f"lost lock at {obj._lock_path} before writing final result"
        )

    tmp_result_dir = _nfs_safe_unique_name(obj._result_dir, name="tmp")

    try:
        save_result_bundle(result, tmp_result_dir)

        if not has_lock():
            raise LockLostError(
                f"lost lock at {obj._lock_path} after writing temporary result"
            )

        tmp_result_dir.rename(obj._result_dir)
        obj._metadata_path.write_text(metadata.to_complete().model_dump_json(indent=2))
        obj.logger.debug("stored result bundle at %s", obj._result_dir)
        return cast(T, load_result_bundle(obj._result_dir))
    except BaseException:
        if tmp_result_dir.exists():
            shutil.rmtree(tmp_result_dir, ignore_errors=True)
        raise


def _write_error_logs[T](objs: Sequence[Furu[T]], exc: BaseException) -> None:
    timestamp = datetime.now().strftime("%y%m%d_%H-%M-%S")
    suffix = secrets.token_hex(4)
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
    error_text = "".join(parts)
    for obj in objs:
        obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
        error_path = obj._internal_furu_dir / f"error-{timestamp}-{suffix}.log"
        error_path.write_text(error_text, encoding="utf-8")


@overload
def load_or_create[T](obj: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def load_or_create[T](objs: Sequence[Furu[T]], *, use_lock: bool = True) -> list[T]: ...


def load_or_create[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    if isinstance(obj_or_objs, Furu):
        objs = [obj_or_objs]
        unwrap = True
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

    if not objs:
        return []

    unique_by_dir: dict[Path, Furu[T]] = {}
    for obj in objs:
        unique_by_dir.setdefault(obj.data_dir, obj)
    unique = list(unique_by_dir.values())

    results_by_dir: dict[Path, T] = {}
    missing: list[Furu[T]] = []

    for obj in unique:
        if obj._result_manifest_path.exists():
            obj.logger.info("cache hit for %s at %s", obj._log_label, obj._result_dir)
            results_by_dir[obj.data_dir] = cast(T, load_result_bundle(obj._result_dir))
        else:
            obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
            missing.append(obj)

    lock_ctx = (
        lock_many([obj._lock_path for obj in missing])
        if use_lock and missing
        else nullcontext()
    )

    with lock_ctx as maybe_has_lock:
        has_lock = maybe_has_lock or (lambda: True)
        pending: list[Furu[T]] = []
        for obj in missing:
            if obj._result_manifest_path.exists():
                obj.logger.info(
                    "cache hit for %s after waiting at %s",
                    obj._log_label,
                    obj._result_dir,
                )
                results_by_dir[obj.data_dir] = cast(
                    T,
                    load_result_bundle(obj._result_dir),
                )
            else:
                pending.append(obj)

        grouped: dict[type[object], list[Furu[T]]] = {}
        for obj in pending:
            grouped.setdefault(type(obj), []).append(obj)

        for group in grouped.values():
            _execute_group(group, has_lock=has_lock, results_by_dir=results_by_dir)

    outputs = [results_by_dir[obj.data_dir] for obj in objs]

    if unwrap:
        objs[0].logger.info("%s.load_or_create() returned", objs[0]._log_label)
        return outputs[0]
    return outputs


def _execute_group[T](
    group: list[Furu[T]],
    *,
    has_lock: HasLock,
    results_by_dir: dict[Path, T],
) -> None:
    log_paths = tuple(obj._log_path for obj in group)
    metadata_by_dir = {obj.data_dir: RunningMetadata.write_for(obj) for obj in group}

    with _scoped_log_files(log_paths):
        logger = group[0].logger
        logger.debug("load_or_create start")
        try:
            match group[0]._furu_create_mode:
                case "batched":
                    logger.debug("running _create_batched()")
                    results = type(group[0])._create_batched(group)
                    logger.debug("_create_batched() returned")
                    if not isinstance(results, list):
                        raise TypeError(
                            f"{type(group[0]).__name__}._create_batched() must return a list"
                        )
                case "single":
                    logger.debug("running sequential _create() fallback")
                    results = [obj._create() for obj in group]
                    logger.debug("sequential _create() fallback returned")
                case _:
                    assert_never(group[0]._furu_create_mode)

            if len(results) != len(group):
                raise TypeError(
                    f"{type(group[0]).__name__} returned {len(results)} results for {len(group)} objects"
                )

            for obj, result in zip(group, results, strict=True):
                loaded_result = _store_result(
                    obj,
                    result,
                    metadata=metadata_by_dir[obj.data_dir],
                    has_lock=has_lock,
                )
                results_by_dir[obj.data_dir] = loaded_result

            logger.debug("load_or_create complete")
        except BaseException as exc:
            logger.exception("load_or_create failed")
            _write_error_logs(group, exc)
            raise
