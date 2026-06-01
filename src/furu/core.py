from __future__ import annotations

import logging
import shutil
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, final

from furu._storage_layout import (
    compute_lock_path_in,
    data_dir_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.config import get_config
from furu.locking import is_active_lock, lock_many
from furu.logging import get_logger
from furu.resources import ResourceRequirements
from furu.result import load_result_bundle
from furu.result.codec import ResultRegistry, _default_result_registry
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import (
    JsonValue,
    _hash_dict_deterministically,
    nfs_safe_unique_name,
    object_id_from_parts,
    fully_qualified_name,
)
from furu.validate import validate_cls

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    from furu.metadata import ArtifactSpec
    from furu.migration import Migration

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


type FuruCreateMode = Literal["single", "batched"] | None


class Furu[T](_FuruDataclassTransform, ABC):
    _furu_create_mode: ClassVar[FuruCreateMode]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Furu:
            return

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
        from furu.execution import _install_create_guards, _resolve_create_mode

        cls._furu_create_mode = _resolve_create_mode(cls)
        _install_create_guards(cls)

    def create(self) -> T:
        raise NotImplementedError(
            f"{type(self).__name__} must implement create() or create_batched()"
        )

    @classmethod
    def create_batched[TFuru: Furu](cls: type[TFuru], objs: list[TFuru]) -> list[T]:
        raise NotImplementedError(
            f"{cls.__name__} must implement create() or create_batched()"
        )

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        return ()

    @cached_property
    def storage_root(self) -> Path:
        return get_config().directories.objects

    @property
    def result_registry(self) -> ResultRegistry:
        return _default_result_registry()

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return None

    @property
    def logger(self) -> logging.Logger:
        return get_logger()

    @final
    @cached_property
    def data_dir(self) -> Path:
        data_dir = data_dir_in(self._base_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @final
    @cached_property
    def object_id(self) -> str:
        return object_id_from_parts(
            fully_qualified_name=self._fully_qualified_name,
            schema_hash=self._artifact_schema_hash,
            artifact_hash=self._artifact_hash,
        )

    @final
    def load_or_create(self, use_lock: bool = True) -> T:
        from furu.execution import load_or_create

        return load_or_create(self, use_lock=use_lock)

    @final
    def load_existing(self) -> T:
        from furu.dependencies import record_dependency_call
        from furu.migration import result_dir_for_loading
        from furu.worker.context import (
            _DependencyNotReady,
            _worker_execution_lease_id,
        )

        record_dependency_call(self)
        if (result_dir := result_dir_for_loading(self)) is not None:
            return cast(T, load_result_bundle(result_dir))
        if _worker_execution_lease_id.get() is not None:
            raise _DependencyNotReady(
                dependencies=[self],
                call_kind="load_existing",
            )
        raise RuntimeError(
            f"{self._log_label}.load_existing() could not find a result. "
            "load_existing() only loads existing results; use load_or_create() to "
            "compute missing results."
        )

    @final
    def status(
        self,
    ) -> Literal[
        "completed", "missing", "running", "failed"
    ]:  # TODO: add queued/waiting state?
        if result_manifest_path_in(self._base_dir).exists():
            return "completed"
        if self.is_migrated():  # TODO: check if the migrated object is in correct state
            return "completed"
        if is_active_lock(compute_lock_path_in(self._base_dir)):
            return "running"
        if self._base_dir.exists():
            return "failed"
        return "missing"

    @final
    def delete(self, mode: Literal["prompt", "force"] = "prompt") -> bool:
        if not self._base_dir.exists():
            return False

        tombstone_path: Path | None = None
        try:
            with lock_many([compute_lock_path_in(self._base_dir)]):
                if not self._base_dir.exists():
                    return False

                if mode == "prompt" and (
                    input(f"Do you want to delete {self._base_dir}? [y/N] ")
                    .strip()
                    .lower()
                    != "y"
                ):
                    return False

                tombstone_path = nfs_safe_unique_name(self._base_dir, name="deleting")
                self._base_dir.rename(tombstone_path)
        except RuntimeError:
            if tombstone_path is None:
                raise

        assert tombstone_path is not None
        shutil.rmtree(tombstone_path)
        return True

    @final
    def migrate(self) -> bool:
        from furu.migration import migrate

        return migrate(self)

    @final
    def is_migrated(self) -> bool:
        return result_link_path_in(self._base_dir).exists()

    @final
    @classmethod
    def from_artifact[TFuru: Furu](cls: type[TFuru], artifact: ArtifactSpec) -> TFuru:
        from furu.serialize import _from_artifact

        return _from_artifact(artifact, cls)

    @final
    @cached_property
    def _artifact_data(  # TODO: make sure this doesn't prevent garbage collection
        self,
    ) -> dict[str, JsonValue]:
        return _to_json(self)  # ty:ignore[invalid-return-type] # TODO: check this or make _to_json return dict[str, JsonValue] or typed value

    @final
    @cached_property
    def _artifact_hash(  # TODO: should this be __hash__?
        self,
    ) -> str:
        return _hash_dict_deterministically(self._artifact_data)

    @final
    @cached_property
    def _schema_data(
        self,
    ) -> JsonValue:
        return _schema_type(type(self), set())

    @final
    @cached_property
    def _artifact_schema_hash(self) -> str:
        return _hash_dict_deterministically(self._schema_data)

    @final
    @cached_property
    def _fully_qualified_name(self) -> str:
        return fully_qualified_name(type(self))

    @final
    @cached_property
    def _base_dir(self) -> Path:
        return (
            self.storage_root
            / Path(*self._fully_qualified_name.split("."))
            / self._artifact_schema_hash
            / self._artifact_hash
        )

    @final
    @cached_property
    def _log_label(self) -> str:
        return (
            f"{type(self).__name__}:"
            + f"{self._artifact_schema_hash[:5]}:"
            + f"{self._artifact_hash[:5]}"
        )
