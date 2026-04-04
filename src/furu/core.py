from __future__ import annotations

import logging
import os
import pickle
import shutil
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from furu.config import config
from furu.locking import Lease, LockLostError, lock
from furu.logging import get_logger
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
        if cls._create is Furu._create:
            raise TypeError(f"{cls.__name__} must implement _create()")
        create_many = cls.__dict__.get("_create_many")
        if create_many is not None and not isinstance(create_many, classmethod):
            raise TypeError(
                f"{cls.__name__}._create_many must be declared as a @classmethod"
            )

    def load_or_create(self, use_lock: bool = True) -> T:
        from furu.execution import load_or_create

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
            return self.load_result()
        raise NotImplementedError(
            "TODO: decide if i should throw or return error value"
        )

    def delete(self, mode: Literal["prompt", "force"] = "prompt") -> bool:
        if not self.data_dir.exists():
            return False

        self._internal_furu_dir.mkdir(exist_ok=True, parents=True)

        tombstone_path: Path | None = None
        try:
            with lock(self._lock_path):
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
    def _create_many(cls, objs: list[Self]) -> list[T]:
        return [obj._create() for obj in objs]

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
    def cache_key(self) -> Path:
        return self.data_dir.resolve(strict=False)

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
    def _lock_path(self) -> Path:
        return self._internal_furu_dir / "compute.lock"

    @cached_property
    def _log_label(self) -> str:
        return f"{type(self).__name__}:{self.schema_hash[:5]}:{self.artifact_hash[:5]}"

    def ensure_private_dir(self) -> None:
        self._internal_furu_dir.mkdir(parents=True, exist_ok=True)

    def load_result(self) -> T:
        with self._result_path.open("rb") as f:
            return pickle.load(f)

    def write_running_metadata(self) -> RunningMetadata:
        metadata = RunningMetadata(
            artifact=self.artifact,
            artifact_hash=self.artifact_hash,
            schema_=self.schema,
            schema_hash=self.schema_hash,
            data_path=self.cache_key,
            started_at=datetime.now(timezone.utc),
        )
        self._metadata_path.write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return metadata

    def commit(
        self,
        result: T,
        *,
        metadata: RunningMetadata,
        lease: Lease | None,
    ) -> None:
        self.ensure_private_dir()
        if lease is not None:
            lease.assert_held(
                f"lost lock at {self._lock_path} before writing final result"
            )

        tmp_result_path = self._result_path.with_suffix(".pkl.tmp")
        with tmp_result_path.open("wb") as f:
            pickle.dump(result, f)
            f.flush()
            os.fsync(f.fileno())

        if lease is not None:
            lease.assert_held(
                f"lost lock at {self._lock_path} after writing temporary result"
            )

        tmp_result_path.rename(self._result_path)
        self._metadata_path.write_text(
            metadata.to_complete().model_dump_json(indent=2),
            encoding="utf-8",
        )
        self.logger.debug("stored result at %s", self._result_path)
