from __future__ import annotations

import logging
import pickle
import shutil
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from furu.config import config
from furu.locking import LockLostError, lock_many
from furu.logging import get_logger
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


type FuruCreateMode = Literal["single", "batched"]


class Furu[T](_FuruDataclassTransform, ABC):
    _furu_create_mode: ClassVar[FuruCreateMode]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Furu:
            return

        validate_cls(cls)
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)
        from furu.execution import resolve_create_mode

        cls._furu_create_mode = resolve_create_mode(cls)

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
            with lock_many([self._lock_path]):
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
    def _create_batched(cls, objs: list[Self]) -> list[T]:
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
    def _lock_path(self) -> Path:
        return self._internal_furu_dir / "compute.lock"

    @cached_property
    def _log_label(self) -> str:
        return f"{type(self).__name__}:{self.schema_hash[:5]}:{self.artifact_hash[:5]}"
