import logging
import inspect
import os
import pickle
import secrets
import shutil
import traceback
from abc import ABC
from collections.abc import Callable, Iterable
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
from furu.locking import LockLostError, lock, lock_many

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


type _CreatorKind = Literal["single", "batched"]
type _Lease = Callable[[], bool] | None


class Furu[T](_FuruDataclassTransform, ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Furu:
            return

        validate_cls(cls)
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)

    def load_or_create(self, *, use_lock: bool = True) -> T:
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


def _overrides_base_method(cls: type[Any], base: type[Any], name: str) -> bool:
    return inspect.getattr_static(cls, name) != inspect.getattr_static(base, name)


def _creator_kind_for_class(cls: type[Furu[Any]]) -> _CreatorKind:
    single = _overrides_base_method(cls, Furu, "_create")
    batched = _overrides_base_method(cls, Furu, "_create_batched")
    if single == batched:
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} must override exactly one of "
            "_create or _create_batched"
        )
    return "single" if single else "batched"


def _artifact_key(obj: Furu[Any]) -> Path:
    return obj.data_dir.resolve()


def _compute_lock_path(obj: Furu[Any]) -> Path:
    return obj._internal_furu_dir / "compute.lock"


def _load_result[T](obj: Furu[T]) -> T:
    with open(obj._result_path, "rb") as f:
        return pickle.load(f)


def _write_failure_diagnostics(obj: Furu[Any], exc: BaseException) -> None:
    obj._internal_furu_dir.mkdir(exist_ok=True, parents=True)

    with _scoped_log_file(obj._log_path):
        obj.logger.error(
            "load_or_create failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    exc_tb = [] if exc.__traceback__ is None else traceback.extract_tb(exc.__traceback__)

    with (
        (
            obj._internal_furu_dir
            / f"error-{datetime.now():%y%m%d_%H-%M-%S}-{secrets.token_hex(4)}.log"
        ).open("a", encoding="utf-8") as f
    ):
        f.write("Traceback (most recent call last):\n")
        f.writelines(
            traceback.format_list(traceback.extract_stack()[:-1] + exc_tb)
        )
        f.writelines(traceback.format_exception_only(type(exc), exc))
        f.write("\n=== Debug Details (with locals) ===\n")
        f.writelines(
            traceback.TracebackException.from_exception(
                exc, capture_locals=True
            ).format(chain=True)
        )


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


def _assert_has_lock(obj: Furu[Any], lease: _Lease, *, phase: str) -> None:
    if lease is not None and not lease():
        raise LockLostError(f"lost lock at {_compute_lock_path(obj)} {phase}")


def _publish_precomputed_one_unlocked[T](
    obj: Furu[T],
    value: T,
    *,
    metadata: RunningMetadata,
    lease: _Lease,
) -> T:
    with _scoped_log_file(obj._log_path):
        completed_metadata = metadata.to_complete()

        _assert_has_lock(obj, lease, phase="before writing final result")

        tmp_result_path = obj._result_path.with_suffix(".tmp.pkl")
        with tmp_result_path.open("wb") as f:
            pickle.dump(value, f)
            f.flush()
            os.fsync(f.fileno())

        _assert_has_lock(obj, lease, phase="after writing temporary result")

        tmp_result_path.rename(obj._result_path)
        obj._metadata_path.write_text(completed_metadata.model_dump_json(indent=2))
        obj.logger.debug("stored result at %s", obj._result_path)
    return value


def _create_and_publish_one_unlocked[T](obj: Furu[T], *, lease: _Lease) -> T:
    with _scoped_log_file(obj._log_path):
        metadata = _write_running_metadata(obj)
        obj.logger.debug("running _create()")
        result = obj._create()
        obj.logger.debug("_create() returned")
        return _publish_precomputed_one_unlocked(
            obj,
            result,
            metadata=metadata,
            lease=lease,
        )


def _prepare_internal_dirs(objs: list[Furu[Any]]) -> None:
    for obj in objs:
        obj._internal_furu_dir.mkdir(exist_ok=True, parents=True)


def _stable_dedupe[T](objs: list[Furu[T]]) -> tuple[list[Furu[T]], list[Path]]:
    unique_by_key: dict[Path, Furu[T]] = {}
    order: list[Path] = []
    for obj in objs:
        key = _artifact_key(obj)
        order.append(key)
        unique_by_key.setdefault(key, obj)
    return list(unique_by_key.values()), order


def _split_cached[T](objs: list[Furu[T]]) -> tuple[dict[Path, T], list[Furu[T]]]:
    cached: dict[Path, T] = {}
    missing: list[Furu[T]] = []
    for obj in objs:
        key = _artifact_key(obj)
        if obj._result_path.exists():
            cached[key] = _load_result(obj)
        else:
            missing.append(obj)
    return cached, missing


def _reconstruct[T](order: list[Path], values: dict[Path, T]) -> list[T]:
    return [values[key] for key in order]


def _validate_same_type[T](objs: list[Furu[T]]) -> type[Furu[T]]:
    first_type = type(objs[0])
    for obj in objs[1:]:
        if type(obj) is not first_type:
            raise TypeError(
                "load_or_create() list inputs must have the exact same concrete type"
            )
    return first_type


def _execute_missing_unlocked[T](
    missing: list[Furu[T]],
    *,
    creator_kind: _CreatorKind,
    lease: _Lease,
    log_failures: bool = True,
) -> dict[Path, T]:
    if creator_kind == "single":
        created: dict[Path, T] = {}
        for obj in missing:
            try:
                created[_artifact_key(obj)] = _create_and_publish_one_unlocked(
                    obj, lease=lease
                )
            except BaseException as exc:
                if log_failures:
                    _write_failure_diagnostics(obj, exc)
                raise
        return created

    metadata_by_key = {
        _artifact_key(obj): _write_running_metadata(obj)
        for obj in missing
    }
    batched_cls = type(missing[0])

    try:
        with _scoped_log_file(missing[0]._log_path):
            missing[0].logger.debug(
                "running _create_batched() for %d items", len(missing)
            )
            batched_results = batched_cls._create_batched(missing)
            missing[0].logger.debug("_create_batched() returned")
    except BaseException as exc:
        if log_failures:
            for obj in missing:
                _write_failure_diagnostics(obj, exc)
        raise

    if len(batched_results) != len(missing):
        exc = ValueError(
            f"{batched_cls.__module__}.{batched_cls.__qualname__}._create_batched "
            f"returned {len(batched_results)} results for {len(missing)} inputs"
        )
        if log_failures:
            for obj in missing:
                _write_failure_diagnostics(obj, exc)
        raise exc

    created: dict[Path, T] = {}
    for obj, value in zip(missing, batched_results, strict=True):
        try:
            created[_artifact_key(obj)] = _publish_precomputed_one_unlocked(
                obj,
                value,
                metadata=metadata_by_key[_artifact_key(obj)],
                lease=lease,
            )
        except BaseException as exc:
            if log_failures:
                _write_failure_diagnostics(obj, exc)
            raise
    return created


def _load_or_create_many[T](
    objs: list[Furu[T]],
    *,
    use_lock: bool,
    log_failures: bool = True,
) -> list[T]:
    cls = _validate_same_type(objs)
    creator_kind = _creator_kind_for_class(cls)

    unique, order = _stable_dedupe(objs)
    cached, missing = _split_cached(unique)
    if not missing:
        return _reconstruct(order, cached)

    _prepare_internal_dirs(missing)

    if not use_lock:
        created = _execute_missing_unlocked(
            missing,
            creator_kind=creator_kind,
            lease=None,
            log_failures=log_failures,
        )
        return _reconstruct(order, {**cached, **created})

    lock_paths = [_compute_lock_path(obj) for obj in missing]
    with lock_many(lock_paths) as lease:
        cached_after_lock, missing_after_lock = _split_cached(unique)
        if not missing_after_lock:
            return _reconstruct(order, cached_after_lock)

        _prepare_internal_dirs(missing_after_lock)
        created = _execute_missing_unlocked(
            missing_after_lock,
            creator_kind=creator_kind,
            lease=lease,
            log_failures=log_failures,
        )
        return _reconstruct(order, {**cached_after_lock, **created})


def _load_or_create_scalar[T](obj: Furu[T], *, use_lock: bool) -> T:
    if obj._result_path.exists():
        obj.logger.info("cache hit for %s at %s", obj._log_label, obj._result_path)
        return _load_result(obj)

    obj._internal_furu_dir.mkdir(exist_ok=True, parents=True)

    try:
        obj.logger.info("calling %s.load_or_create()", obj._log_label)
        with _scoped_log_file(obj._log_path):
            obj.logger.debug("load_or_create start")
            result = _load_or_create_many(
                [obj],
                use_lock=use_lock,
                log_failures=False,
            )[0]
            obj.logger.debug("load_or_create complete")
        obj.logger.info("%s.load_or_create() returned", obj._log_label)
        return result
    except BaseException as exc:
        _write_failure_diagnostics(obj, exc)
        raise


@overload
def load_or_create[T](obj: Furu[T], *, use_lock: bool = True) -> T: ...


@overload
def load_or_create[T](objs: list[Furu[T]], *, use_lock: bool = True) -> list[T]: ...


def load_or_create[T](
    obj_or_objs: Furu[T] | Iterable[Furu[T]],
    *,
    use_lock: bool = True,
) -> T | list[T]:
    if isinstance(obj_or_objs, Furu):
        return _load_or_create_scalar(obj_or_objs, use_lock=use_lock)

    objs = list(obj_or_objs)
    if not objs:
        return []

    return _load_or_create_many(objs, use_lock=use_lock)
