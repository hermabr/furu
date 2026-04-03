import os
import pickle
import secrets
import shutil
import logging
import traceback
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Self, TypeVar, cast, overload

from furu.config import config
from furu.logging import _scoped_log_files, get_logger
from furu.locking import LockLostError, lock, lock_many
from furu.metadata import RunningMetadata
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import (
    JsonValue,
    _hash_dict_deterministically,
    _nfs_safe_unique_name,
    fully_qualified_name,
)
from furu.validate import get_create_mode, validate_cls

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


_ACTIVE_LOAD_KEYS: ContextVar[frozenset[str]] = ContextVar(
    "furu_active_load_keys", default=frozenset()
)
T = TypeVar("T")


class Furu[T](_FuruDataclassTransform):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Furu:
            return

        validate_cls(cls)
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)

    def load_or_create(self, use_lock: bool = True) -> T:
        return load_or_create(self, use_lock=use_lock)

    def status(
        self,
    ) -> Literal[
        "completed", "missing", "running", "failed"
    ]:  # TODO: add queued/waiting state?
        if self.is_completed():
            return "completed"
        raise NotImplementedError("TODO")

    def is_completed(
        self,
    ) -> bool:  # TODO: maybe this should check the is self.status is completed? (in that case status cant check if self.is_completed)
        return self._result_path.exists()

    def try_load(self) -> T:  # TODO: make a better name for this
        if self._result_path.exists():
            with self._result_path.open("rb") as f:
                return pickle.load(f)
        raise NotImplementedError(
            "TODO: decide if i should throw or return error value"
        )

    def delete(self, mode: Literal["prompt", "force"] = "prompt") -> bool:
        if not self.data_dir.exists():
            return False

        self._internal_furu_dir.mkdir(exist_ok=True, parents=True)

        tombstone_path: Path | None = None
        try:
            with lock(self._internal_furu_dir / "compute.lock"):
                if not self.data_dir.exists():
                    return False

                if mode == "prompt" and (
                    input(f"Do you want to delete {self.data_dir}? [y/N] ")
                    .strip()
                    .lower()
                    != "y"
                ):
                    return False

                tombstone_path = _nfs_safe_unique_name(self.data_dir, name="deleting")
                self.data_dir.rename(tombstone_path)
        except LockLostError:
            if tombstone_path is None:
                raise

        assert tombstone_path is not None
        shutil.rmtree(tombstone_path)
        return True

    @property
    def _result_path(self) -> Path:
        return self.data_dir / "result.pkl"

    @property
    def logger(self) -> logging.Logger:
        return get_logger()

    def _create(self) -> T:
        raise NotImplementedError(
            f"{fully_qualified_name(type(self))} must override _create() or "
            "_create_batched()"
        )

    @classmethod
    def _create_batched(cls, objs: Sequence[Self]) -> list[T]:
        raise NotImplementedError(
            f"{fully_qualified_name(cls)} must override _create() or "
            "_create_batched()"
        )

    @cached_property
    def artifact(  # TODO: make sure this doesn't prevent garbage collection
        self,
    ) -> JsonValue:
        return _to_json(self)

    @classmethod
    def from_json(cls) -> Self:
        raise NotImplementedError("TODO")

    @cached_property
    def artifact_hash(  # TODO: should this be __hash__?
        self,
    ) -> str:
        return _hash_dict_deterministically(self.artifact)

    @cached_property
    def schema(
        self,
    ) -> JsonValue:
        return _schema_type(type(self), set())

    @cached_property
    def schema_hash(self) -> str:
        return _hash_dict_deterministically(self.schema)

    @cached_property  # TODO: decide if something like this should be cached_property or simply property
    def data_dir(self) -> Path:
        return (
            config.directories.data
            / Path(*fully_qualified_name(type(self)).split("."))
            / self.schema_hash
            / self.artifact_hash
        )

    @cached_property
    def _internal_furu_dir(self) -> Path:
        return self.data_dir / ".furu"

    @cached_property
    def _metadata_path(self) -> Path:
        return self._internal_furu_dir / "metadata.json"

    @cached_property
    def _log_path(self) -> Path:
        return self._internal_furu_dir / "run.log"

    @cached_property
    def _log_label(self) -> str:
        return f"{type(self).__name__}:{self.schema_hash[:5]}:{self.artifact_hash[:5]}"


@dataclass(slots=True)
class _UniqueItem:
    obj: Furu[Any]
    data_dir_key: str
    input_indices: list[int]


def _canonical_data_dir_key(obj: Furu[Any]) -> str:
    return os.fspath(obj.data_dir.resolve())


def _load_cached_result(obj: Furu[Any], *, after_wait: bool = False) -> Any:
    with _scoped_log_files((obj._log_path,)):
        if after_wait:
            obj.logger.info(
                "cache hit for %s after waiting at %s",
                obj._log_label,
                obj._result_path,
            )
        else:
            obj.logger.info(
                "cache hit for %s at %s",
                obj._log_label,
                obj._result_path,
            )
        with obj._result_path.open("rb") as f:
            result = pickle.load(f)
        obj.logger.debug("loaded result from %s", obj._result_path)
        return result


def _write_running_metadata(
    obj: Furu[Any], *, started_at: datetime | None = None
) -> RunningMetadata:
    metadata = RunningMetadata(
        artifact=obj.artifact,
        artifact_hash=obj.artifact_hash,
        schema_=obj.schema,
        schema_hash=obj.schema_hash,
        data_path=obj.data_dir.resolve(),
        started_at=started_at or datetime.now(timezone.utc),
    )

    with _scoped_log_files((obj._log_path,)):
        obj.logger.debug("load_or_create start")
        obj._metadata_path.write_text(metadata.model_dump_json(indent=2))
        obj.logger.debug("wrote running metadata to %s", obj._metadata_path)

    return metadata


def _ensure_has_lock(
    obj: Furu[Any],
    has_lock: Callable[[], bool] | None,
    *,
    phase: Literal["before writing final result", "after writing temporary result"],
) -> None:
    if has_lock is not None and not has_lock():
        raise LockLostError(
            f"lost lock at {obj._internal_furu_dir / 'compute.lock'} {phase}"
        )


def _store_result(
    obj: Furu[Any],
    metadata: RunningMetadata,
    result: Any,
    *,
    has_lock: Callable[[], bool] | None,
) -> None:
    completed_metadata = metadata.to_complete()

    with _scoped_log_files((obj._log_path,)):
        _ensure_has_lock(obj, has_lock, phase="before writing final result")

        tmp_result_path = obj._result_path.with_suffix(".tmp.pkl")
        with tmp_result_path.open("wb") as f:
            pickle.dump(result, f)
            f.flush()
            os.fsync(f.fileno())

        _ensure_has_lock(obj, has_lock, phase="after writing temporary result")

        tmp_result_path.rename(obj._result_path)
        obj._metadata_path.write_text(completed_metadata.model_dump_json(indent=2))
        obj.logger.debug("stored result at %s", obj._result_path)
        obj.logger.debug("load_or_create complete")


def _record_failure(obj: Furu[Any], exc: BaseException) -> None:
    obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)

    with _scoped_log_files((obj._log_path,)):
        obj.logger.exception("load_or_create failed")

    error_path = (
        obj._internal_furu_dir
        / f"error-{datetime.now():%y%m%d_%H-%M-%S}-{secrets.token_hex(4)}.log"
    )
    with error_path.open("a", encoding="utf-8") as f:
        f.write("Traceback (most recent call last):\n")
        f.writelines(
            traceback.format_list(
                traceback.extract_stack()[:-1] + traceback.extract_tb(exc.__traceback__)
            )
        )
        f.writelines(traceback.format_exception_only(type(exc), exc))
        f.write("\n=== Debug Details (with locals) ===\n")
        f.writelines(
            traceback.TracebackException.from_exception(
                exc, capture_locals=True
            ).format(chain=True)
        )


def _normalize_batch_input(
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
) -> tuple[list[Furu[Any]], bool]:
    if isinstance(obj_or_objs, Furu):
        return [cast(Furu[Any], obj_or_objs)], True

    objs = list(obj_or_objs)
    for obj in objs:
        if not isinstance(obj, Furu):
            raise TypeError(
                "load_or_create() expects a Furu object or a sequence of Furu objects"
            )
    return cast(list[Furu[Any]], objs), False


def _dedupe_by_data_dir(objs: Sequence[Furu[Any]]) -> list[_UniqueItem]:
    unique_by_key: dict[str, _UniqueItem] = {}
    ordered_items: list[_UniqueItem] = []

    for index, obj in enumerate(objs):
        data_dir_key = _canonical_data_dir_key(obj)
        item = unique_by_key.get(data_dir_key)
        if item is None:
            item = _UniqueItem(obj=obj, data_dir_key=data_dir_key, input_indices=[index])
            unique_by_key[data_dir_key] = item
            ordered_items.append(item)
        else:
            item.input_indices.append(index)

    return ordered_items


def _plan_execution(
    unique_items: Sequence[_UniqueItem],
) -> tuple[dict[str, Any], list[_UniqueItem]]:
    results_by_key: dict[str, Any] = {}
    pending_items: list[_UniqueItem] = []

    for item in unique_items:
        if item.obj._result_path.exists():
            results_by_key[item.data_dir_key] = _load_cached_result(item.obj)
            continue

        item.obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
        pending_items.append(item)

    return results_by_key, pending_items


@contextmanager
def _guard_same_thread_reentry(pending_items: Sequence[_UniqueItem]):
    pending_keys = tuple(
        item.data_dir_key for item in pending_items if not item.obj._result_path.exists()
    )
    active_keys = _ACTIVE_LOAD_KEYS.get()
    reentrant_keys = [key for key in pending_keys if key in active_keys]
    if reentrant_keys:
        raise RuntimeError(
            "same-thread load_or_create reentry detected for unresolved object(s): "
            + ", ".join(sorted(dict.fromkeys(reentrant_keys)))
        )

    token = _ACTIVE_LOAD_KEYS.set(active_keys.union(pending_keys))
    try:
        yield
    finally:
        _ACTIVE_LOAD_KEYS.reset(token)


def _group_pending_items(pending_items: Sequence[_UniqueItem]) -> list[list[_UniqueItem]]:
    grouped_items: dict[type[Furu[Any]], list[_UniqueItem]] = {}
    for item in pending_items:
        grouped_items.setdefault(type(item.obj), []).append(item)
    return list(grouped_items.values())


def _recheck_pending_items_after_lock(
    pending_items: Sequence[_UniqueItem], results_by_key: dict[str, Any]
) -> list[_UniqueItem]:
    still_pending: list[_UniqueItem] = []
    for item in pending_items:
        if item.obj._result_path.exists():
            results_by_key[item.data_dir_key] = _load_cached_result(
                item.obj, after_wait=True
            )
        else:
            still_pending.append(item)
    return still_pending


def _run_single_item(
    item: _UniqueItem,
    *,
    results_by_key: dict[str, Any],
    has_lock: Callable[[], bool] | None,
) -> None:
    metadata = _write_running_metadata(item.obj)

    try:
        with _scoped_log_files((item.obj._log_path,)):
            item.obj.logger.debug("running _create()")
            result = item.obj._create()
            item.obj.logger.debug("_create() returned")
        _store_result(item.obj, metadata, result, has_lock=has_lock)
    except BaseException as exc:
        _record_failure(item.obj, exc)
        raise

    results_by_key[item.data_dir_key] = result


def _run_batched_group(
    group: Sequence[_UniqueItem],
    *,
    results_by_key: dict[str, Any],
    has_lock: Callable[[], bool] | None,
) -> None:
    if not group:
        return

    started_at = datetime.now(timezone.utc)
    metadata_by_key = {
        item.data_dir_key: _write_running_metadata(item.obj, started_at=started_at)
        for item in group
    }
    cls = type(group[0].obj)
    objs = [cast(Any, item.obj) for item in group]

    try:
        with _scoped_log_files([item.obj._log_path for item in group]):
            group[0].obj.logger.debug(
                "running %s._create_batched() for %d objects",
                cls.__name__,
                len(group),
            )
            raw_results = cls._create_batched(objs)
            group[0].obj.logger.debug("%s._create_batched() returned", cls.__name__)
        if not isinstance(raw_results, list):
            raise TypeError(
                f"{cls.__module__}.{cls.__qualname__}._create_batched() must return a list"
            )
        if len(raw_results) != len(group):
            raise ValueError(
                f"{cls.__module__}.{cls.__qualname__}._create_batched() returned "
                f"{len(raw_results)} result(s) for {len(group)} object(s)"
            )
    except BaseException as exc:
        for item in group:
            _record_failure(item.obj, exc)
        raise

    for item, result in zip(group, raw_results, strict=True):
        try:
            _store_result(
                item.obj,
                metadata_by_key[item.data_dir_key],
                result,
                has_lock=has_lock,
            )
        except BaseException as exc:
            _record_failure(item.obj, exc)
            raise
        results_by_key[item.data_dir_key] = result


def _run_pending_items(
    pending_items: Sequence[_UniqueItem],
    *,
    results_by_key: dict[str, Any],
    has_lock: Callable[[], bool] | None,
) -> None:
    for group in _group_pending_items(pending_items):
        mode = get_create_mode(type(group[0].obj))
        if mode == "single":
            for item in group:
                _run_single_item(
                    item,
                    results_by_key=results_by_key,
                    has_lock=has_lock,
                )
        else:
            _run_batched_group(
                group,
                results_by_key=results_by_key,
                has_lock=has_lock,
            )


def _materialize_outputs(
    unique_items: Sequence[_UniqueItem],
    *,
    results_by_key: dict[str, Any],
    output_size: int,
) -> list[Any]:
    outputs: list[Any] = [None] * output_size
    for item in unique_items:
        result = results_by_key[item.data_dir_key]
        for index in item.input_indices:
            outputs[index] = result
    return outputs


@overload
def load_or_create(obj: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def load_or_create(objs: Sequence[Furu[T]], *, use_lock: bool = True) -> list[T]: ...


def load_or_create(
    obj_or_objs: Furu[T] | Sequence[Furu[T]], *, use_lock: bool = True
) -> T | list[T]:
    objs, is_single_input = _normalize_batch_input(obj_or_objs)
    if not objs:
        return []

    if is_single_input:
        objs[0].logger.info("calling %s.load_or_create()", objs[0]._log_label)

    unique_items = _dedupe_by_data_dir(objs)
    results_by_key, pending_items = _plan_execution(unique_items)
    lock_paths = [item.obj._internal_furu_dir / "compute.lock" for item in pending_items]

    with _guard_same_thread_reentry(pending_items):
        with (
            lock_many(lock_paths) if use_lock and lock_paths else nullcontext(None)
        ) as has_lock:
            if has_lock is not None:
                pending_items = _recheck_pending_items_after_lock(
                    pending_items, results_by_key
                )
            if pending_items:
                _run_pending_items(
                    pending_items,
                    results_by_key=results_by_key,
                    has_lock=has_lock,
                )

    outputs = _materialize_outputs(
        unique_items,
        results_by_key=results_by_key,
        output_size=len(objs),
    )

    if is_single_input:
        objs[0].logger.info("%s.load_or_create() returned", objs[0]._log_label)
        return cast(T, outputs[0])
    return cast(list[T], outputs)
