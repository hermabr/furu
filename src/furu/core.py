import os
import pickle
import secrets
import shutil
import logging
import inspect
import traceback
from abc import ABC
from collections.abc import Callable, Iterable
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    overload,
)

from furu.config import config
from furu.logging import _scoped_log_file, get_logger
from furu.locking import LockLostError, lock

# from furu.locking import run_with_lease_and_pickle_result
from furu.metadata import RunningMetadata
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import (
    JsonValue,
    _hash_dict_deterministically,
    _nfs_safe_unique_name,
    fully_qualified_name,
)
from furu.validate import validate_cls

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


class Furu[T](_FuruDataclassTransform, ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Furu:
            return

        validate_cls(cls)
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)

    def load_or_create(self, use_lock: bool = True) -> T:
        mode = _create_mode(type(self))
        if mode == "scalar":
            return _load_or_create_scalar(self, use_lock=use_lock)
        return _load_or_create_many([self], use_lock=use_lock)[0]

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
            # TODO: validation that its up to date and valid
            with open(self._result_path, "rb") as f:
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
        raise NotImplementedError("TODO")

    @classmethod
    def _create_batched(cls, items: list[Self]) -> list[T]:
        raise NotImplementedError("TODO")

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


def _create_mode(cls: type[Furu[Any]]) -> Literal["scalar", "batch"]:
    scalar_impl = inspect.getattr_static(cls, "_create")
    batch_impl = inspect.getattr_static(cls, "_create_batched")
    base_scalar = inspect.getattr_static(Furu, "_create")
    base_batch = inspect.getattr_static(Furu, "_create_batched")

    has_scalar = scalar_impl is not base_scalar
    has_batch = batch_impl is not base_batch

    if has_scalar == has_batch:
        raise TypeError(
            f"{cls.__qualname__} must implement exactly one of "
            "_create or _create_batched"
        )

    return "scalar" if has_scalar else "batch"


def _validate_same_concrete_furu_type(items: list[Furu[Any]]) -> type[Furu[Any]]:
    cls = type(items[0])
    if not isinstance(items[0], Furu):
        raise TypeError("load_or_create() expects Furu instances")

    for item in items[1:]:
        if not isinstance(item, Furu):
            raise TypeError("load_or_create() expects Furu instances")
        if type(item) is not cls:
            raise TypeError("load_or_create() requires the same concrete Furu type")

    return cls


def _storage_identity(obj: Furu[Any]) -> Path:
    return obj.data_dir.resolve()


def _ensure_internal_furu_dir(obj: Furu[Any]) -> None:
    obj._internal_furu_dir.mkdir(exist_ok=True, parents=True)


def _try_load_completed_result(
    obj: Furu[Any], *, after_wait: bool = False
) -> tuple[bool, Any]:
    if not obj._result_path.exists():
        return False, None

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
        return True, pickle.load(f)


def _load_completed(
    items: list[Furu[Any]], *, after_wait: bool = False
) -> dict[Path, Any]:
    results: dict[Path, Any] = {}
    for obj in items:
        found, result = _try_load_completed_result(obj, after_wait=after_wait)
        if found:
            results[_storage_identity(obj)] = result
    return results


def _write_running_metadata(obj: Furu[Any]) -> RunningMetadata:
    metadata = RunningMetadata(
        artifact=obj.artifact,
        artifact_hash=obj.artifact_hash,
        schema_=obj.schema,
        schema_hash=obj.schema_hash,
        data_path=obj.data_dir.resolve(),
        started_at=datetime.now(timezone.utc),
    )
    obj._metadata_path.write_text(metadata.model_dump_json(indent=2))
    return metadata


def _commit_result(
    obj: Furu[Any],
    result: Any,
    *,
    metadata: RunningMetadata,
    has_lock: Callable[[], bool] | None = None,
) -> None:
    completed_metadata = metadata.to_complete()
    lock_path = obj._internal_furu_dir / "compute.lock"

    if has_lock is not None and not has_lock():
        raise LockLostError(f"lost lock at {lock_path} before writing final result")

    tmp_result_path = obj._result_path.with_suffix(".tmp.pkl")
    with tmp_result_path.open("wb") as f:
        pickle.dump(result, f)
        f.flush()  # TODO: Do i need this and the os.fsync?
        os.fsync(f.fileno())

    if has_lock is not None and not has_lock():
        raise LockLostError(
            f"lost lock at {lock_path} after writing temporary result"
        )

    tmp_result_path.rename(obj._result_path)
    obj._metadata_path.write_text(completed_metadata.model_dump_json(indent=2))
    obj.logger.debug("stored result at %s", obj._result_path)


def _handle_load_or_create_failure(obj: Furu[Any], exc: BaseException) -> None:
    with _scoped_log_file(obj._log_path):
        obj.logger.exception("load_or_create failed")

    with (  # TODO: log this to the regular log file
        (
            obj._internal_furu_dir
            / f"error-{datetime.now():%y%m%d_%H-%M-%S}-{secrets.token_hex(4)}.log"  # TODO: make this part of the regular error
        ).open("a", encoding="utf-8") as f
    ):
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


def _load_or_create_scalar(obj: Furu[Any], *, use_lock: bool) -> Any:
    found, result = _try_load_completed_result(obj)
    if found:
        return result

    _ensure_internal_furu_dir(obj)

    try:
        obj.logger.info("calling %s.load_or_create()", obj._log_label)
        with _scoped_log_file(obj._log_path):
            obj.logger.debug("load_or_create start")

            with (
                lock(obj._internal_furu_dir / "compute.lock")
                if use_lock
                else nullcontext(None)
            ) as has_lock:
                found, result = _try_load_completed_result(obj, after_wait=True)
                if found:
                    return result

                metadata = _write_running_metadata(obj)
                obj.logger.debug("running _create()")
                result = obj._create()
                obj.logger.debug("_create() returned")
                _commit_result(obj, result, metadata=metadata, has_lock=has_lock)
                obj.logger.debug("load_or_create complete")

        obj.logger.info("%s.load_or_create() returned", obj._log_label)
    except BaseException as exc:
        _handle_load_or_create_failure(obj, exc)
        raise

    return result


def _materialize_batched_missing(
    missing: list[Furu[Any]],
    results: dict[Path, Any],
    *,
    use_lock: bool,
) -> None:
    if not missing:
        return

    for obj in missing:
        _ensure_internal_furu_dir(obj)

    cls = type(missing[0])
    has_lock_by_key: dict[Path, Callable[[], bool]] = {}
    metadata_by_key: dict[Path, RunningMetadata] = {}
    started_items: list[Furu[Any]] = []
    committed_keys: set[Path] = set()

    try:
        with ExitStack() as stack:
            if use_lock:
                for obj in sorted(missing, key=lambda item: str(_storage_identity(item))):
                    has_lock_by_key[_storage_identity(obj)] = stack.enter_context(
                        lock(obj._internal_furu_dir / "compute.lock")
                    )

                results.update(_load_completed(missing, after_wait=True))

            still_missing = [
                obj for obj in missing if _storage_identity(obj) not in results
            ]
            if not still_missing:
                return

            batch_size = len(still_missing)
            for batch_index, obj in enumerate(still_missing):
                key = _storage_identity(obj)
                with _scoped_log_file(obj._log_path):
                    obj.logger.info("calling %s.load_or_create()", obj._log_label)
                    obj.logger.debug("load_or_create start")
                    obj.logger.debug(
                        "running _create_batched() for batch item %s of %s",
                        batch_index + 1,
                        batch_size,
                    )
                    metadata_by_key[key] = _write_running_metadata(obj)
                started_items.append(obj)

            with _scoped_log_file(still_missing[0]._log_path):
                batch_results = cls._create_batched(still_missing)

            if len(batch_results) != batch_size:
                raise ValueError(
                    f"{cls.__qualname__}._create_batched() returned "
                    f"{len(batch_results)} results for {batch_size} items"
                )

            for obj, result in zip(still_missing, batch_results):
                key = _storage_identity(obj)
                with _scoped_log_file(obj._log_path):
                    obj.logger.debug("_create_batched() returned")
                    _commit_result(
                        obj,
                        result,
                        metadata=metadata_by_key[key],
                        has_lock=has_lock_by_key.get(key),
                    )
                    obj.logger.debug("load_or_create complete")
                obj.logger.info("%s.load_or_create() returned", obj._log_label)
                results[key] = result
                committed_keys.add(key)
    except BaseException as exc:
        for obj in started_items:
            if _storage_identity(obj) in committed_keys:
                continue
            _handle_load_or_create_failure(obj, exc)
        raise


def _load_or_create_many(items: list[Furu[Any]], *, use_lock: bool) -> list[Any]:
    cls = _validate_same_concrete_furu_type(items)
    mode = _create_mode(cls)

    keys_in_order = [_storage_identity(obj) for obj in items]

    unique_by_key: dict[Path, Furu[Any]] = {}
    for obj, key in zip(items, keys_in_order):
        unique_by_key.setdefault(key, obj)
    unique_items = list(unique_by_key.values())

    results = _load_completed(unique_items)
    missing = [obj for obj in unique_items if _storage_identity(obj) not in results]

    if mode == "scalar":
        for obj in missing:
            results[_storage_identity(obj)] = _load_or_create_scalar(
                obj, use_lock=use_lock
            )
    else:
        _materialize_batched_missing(missing, results, use_lock=use_lock)

    return [results[key] for key in keys_in_order]


@overload
def load_or_create[T](obj_or_items: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def load_or_create[T](
    obj_or_items: Iterable[Furu[T]], *, use_lock: bool = True
) -> list[T]: ...


def load_or_create(
    obj_or_items: Furu[Any] | Iterable[Furu[Any]], *, use_lock: bool = True
) -> Any | list[Any]:
    if isinstance(obj_or_items, Furu):
        return obj_or_items.load_or_create(use_lock=use_lock)

    items = list(obj_or_items)
    if not items:
        return []

    return _load_or_create_many(items, use_lock=use_lock)
