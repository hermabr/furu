from __future__ import annotations

import logging
import shutil
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from furu.config import config
from furu.locking import LockLostError, lock_many
from furu.logging import get_logger
from furu.migration import Migration, _resolve_result_manifest_path
from furu.result import load_result_bundle
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import (
    JsonValue,
    _hash_dict_deterministically,
    fully_qualified_name,
    nfs_safe_unique_name,
)
from furu.validate import validate_cls

if TYPE_CHECKING:
    from furu.metadata import ArtifactMetadata
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

        if "data_dir" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override data_dir; override storage_root "
                "instead"
            )

        annotations = getattr(cls, "__annotations__", {})
        for name, value in cls.__dict__.items():
            if (
                not (name.startswith("__") and name.endswith("__"))
                and name not in annotations
                and not callable(value)
                and not isinstance(value, (classmethod, property, cached_property))
            ):
                raise TypeError(f"{cls.__name__}.{name} must have a type annotation")

        validate_cls(cls)
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)
        from furu.execution import _resolve_create_mode

        cls._furu_create_mode = _resolve_create_mode(cls)

    def load_or_create(self, use_lock: bool = True) -> T:
        from furu.execution import load_or_create

        return load_or_create(self, use_lock=use_lock)

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        return ()

    @classmethod
    def from_artifact[TFuru: Furu](
        cls: type[TFuru], artifact: ArtifactMetadata
    ) -> TFuru:
        from furu.serialize import _from_artifact

        return _from_artifact(artifact, cls)

    def status(
        self,
    ) -> Literal[
        "completed", "missing", "running", "failed"
    ]:  # TODO: add queued/waiting state?
        if self._result_manifest_path is not None:
            return "completed"
        if self._lock_path.exists():
            return "running"
        if self._metadata_path.exists() or self._result_dir.exists():
            return "failed"
        return "missing"

    def try_load(self) -> T:  # TODO: make a better name for this
        if (result_manifest_path := self._result_manifest_path) is not None:
            return cast(T, load_result_bundle(result_manifest_path.parent))
        raise NotImplementedError(
            "TODO: decide if i should throw or return error value"
        )

    @property
    def _result_manifest_path(self) -> Path | None:
        return _resolve_result_manifest_path(self)

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

                tombstone_path = nfs_safe_unique_name(self.data_dir, name="deleting")
                self.data_dir.rename(tombstone_path)
        except LockLostError:
            if tombstone_path is None:
                raise

        assert tombstone_path is not None
        shutil.rmtree(tombstone_path)
        return True

    @property
    def _result_dir(self) -> Path:
        return self.data_dir / "result"

    @property
    def _own_result_manifest_path(self) -> Path:
        return self._result_dir / "manifest.json"

    @property
    def logger(self) -> logging.Logger:
        return get_logger()

    def _create(self) -> T:
        raise NotImplementedError("TODO")

    @classmethod
    def _create_batched[TFuru: Furu](cls: type[TFuru], objs: list[TFuru]) -> list[T]:
        raise NotImplementedError("TODO")

    @cached_property
    def artifact_data(  # TODO: make sure this doesn't prevent garbage collection
        self,
    ) -> JsonValue:
        return _to_json(self)

    @cached_property
    def artifact_hash(  # TODO: should this be __hash__?
        self,
    ) -> str:
        return _hash_dict_deterministically(self.artifact_data)

    @cached_property
    def schema(
        self,
    ) -> JsonValue:
        return _schema_type(type(self), set())

    @cached_property
    def artifact_schema_hash(self) -> str:
        return _hash_dict_deterministically(self.schema)

    @cached_property
    def _fully_qualified_name(self) -> str:
        return fully_qualified_name(type(self))

    @cached_property
    def object_id(self) -> str:
        return (
            f"{self._fully_qualified_name}:"
            + f"{self.artifact_schema_hash}:"
            + f"{self.artifact_hash}"
        )

    @cached_property
    def storage_root(self) -> Path:
        return config.directories.data

    @cached_property
    def data_dir(self) -> Path:
        return (
            self.storage_root
            / Path(*self._fully_qualified_name.split("."))
            / self.artifact_schema_hash
            / self.artifact_hash
        )

    @cached_property
    def _internal_furu_dir(self) -> Path:
        return self.data_dir / ".furu"

    @cached_property
    def _metadata_path(self) -> Path:
        return self._internal_furu_dir / "metadata.json"

    @cached_property
    def _result_link_path(self) -> Path:
        return self._internal_furu_dir / "result-link.json"

    @cached_property
    def _log_path(self) -> Path:
        return self._internal_furu_dir / "run.log"

    @cached_property
    def _lock_path(self) -> Path:
        return self._internal_furu_dir / "compute.lock"

    @cached_property
    def _log_label(self) -> str:
        return (
            f"{type(self).__name__}:"
            + f"{self.artifact_schema_hash[:5]}:"
            + f"{self.artifact_hash[:5]}"
        )
