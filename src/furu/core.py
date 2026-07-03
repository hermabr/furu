from __future__ import annotations

import json
import logging
import shutil
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from inspect import get_annotations
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TypeAlias,
    cast,
    final,
)

from furu._declared_types import declared_result_type
from furu.config import get_config
from furu.explain import ExplainDepth, explain as _explain
from furu.locking import LockError, is_active_lock, lock
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.resources import ResourceRequirements
from furu.result.bundle import load_result_bundle
from furu.result.codec import Codec
from furu.serializer.artifact import to_json as _to_json
from furu.serializer.registry import Serializer
from furu.serializer.schema import schema_type as _schema_type
from furu.storage._layout import (
    compute_lock_path_in,
    data_dir_in,
    metadata_path_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.storage.directory import SpecDirectory
from furu.utils import (
    JsonValue,
    _hash_dict_deterministically,
    fully_qualified_name,
    nfs_safe_unique_name,
    object_id_from_parts,
)
from furu.validate import validate_cls

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    from furu.migration import Migration

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


class Missing(Exception):
    pass


SpecCreateMode: TypeAlias = Literal["single", "batched"] | None
_FURU_CLASS_OPTIONS = frozenset({"max_workers"})
_RESERVED_FIELD_NAMES = frozenset(
    {
        "create",
        "create_batched",
        "metadata",
        "status",
        "directory",
        "explain",
        "load_existing",
        "delete",
        "migrate",
        "migrations",
        "provenance",
    }
)


class Spec[T](_FuruDataclassTransform, ABC):
    _furu_create_mode: ClassVar[SpecCreateMode]
    max_workers: ClassVar[int | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Spec:
            return

        annotations = get_annotations(cls, eval_str=False)
        if reserved := _RESERVED_FIELD_NAMES & annotations.keys():
            raise TypeError(
                f"{cls.__name__} declares field(s) {sorted(reserved)} that shadow "
                f"the Spec verb surface; reserved names: "
                f"{sorted(_RESERVED_FIELD_NAMES)}"
            )
        if _FURU_CLASS_OPTIONS & annotations.keys():
            annotations = {
                name: value
                for name, value in annotations.items()
                if name not in _FURU_CLASS_OPTIONS
            }
            cls.__annotations__ = annotations
        for name, value in cls.__dict__.items():
            if (
                not (name.startswith("__") and name.endswith("__"))
                and name not in _FURU_CLASS_OPTIONS
                and name not in annotations
                and not callable(value)
                and not isinstance(value, (classmethod, property, cached_property))
            ):
                raise TypeError(f"{cls.__name__}.{name} must have a type annotation")

        validate_cls(cls)
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)
        from furu.execution.load_or_create import (
            _install_create_dispatchers,
            _resolve_create_mode,
        )

        cls._furu_create_mode = _resolve_create_mode(cls)
        _install_create_dispatchers(cls)

    def create(self) -> T:
        from furu.execution.load_or_create import _load_or_create

        return _load_or_create(self)

    @classmethod
    def create_batched[TSpec: Spec](cls: type[TSpec], objs: list[TSpec]) -> list[T]:
        raise NotImplementedError(
            f"{cls.__name__} must implement create() or create_batched()"
        )

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        return ()

    @cached_property
    def storage_root(self) -> Path:
        return get_config().run_directories.objects

    @final
    @cached_property
    def _storage_root(self) -> Path:
        config = get_config()
        if config.debug_mode:
            return config.run_directories.objects
        return self.storage_root

    @property
    def result_codecs(self) -> tuple[type[Codec], ...]:
        return ()

    @property
    def artifact_serializers(self) -> tuple[type[Serializer], ...]:
        return ()

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        return None

    @property
    def logger(self) -> logging.Logger:
        return get_logger()

    @final
    @cached_property
    def directory(self) -> SpecDirectory:
        return SpecDirectory(self._base_dir)

    @final
    @cached_property
    def object_id(self) -> str:
        return object_id_from_parts(
            fully_qualified_name=self._fully_qualified_name,
            schema_hash=self._artifact_schema_hash,
            artifact_hash=self._artifact_hash,
        )

    @final
    def _load_or_create(self, use_lock: bool = True) -> T:
        from furu.execution.load_or_create import _load_or_create

        return _load_or_create(self, use_lock=use_lock)

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
            return cast(
                T,
                load_result_bundle(
                    result_dir,
                    data_dir=data_dir_in(self._base_dir),
                    declared_type=declared_result_type(type(self)),
                ),
            )
        if _worker_execution_lease_id.get() is not None:
            raise _DependencyNotReady(
                dependencies=[self],
                call_kind="load_existing",
            )
        raise Missing(
            f"{self._log_label}.load_existing() could not find a result. "
            "load_existing() only loads existing results; use create() to compute "
            "missing results."
        )

    @final
    @property
    def status(self) -> Literal["missing", "running", "failed", "done"]:
        if result_manifest_path_in(self._base_dir).exists():
            return "done"
        if self.is_migrated():  # TODO: check if the migrated object is in correct state
            return "done"
        if is_active_lock(compute_lock_path_in(self._base_dir)):
            return "running"
        if self._base_dir.exists():
            return "failed"
        return "missing"

    @final
    def explain(self, depth: ExplainDepth = 0) -> str:
        return _explain(self, depth=depth)

    @final
    def delete(self, mode: Literal["prompt", "force"] = "prompt") -> bool:
        if not self._base_dir.exists():
            return False

        tombstone_path: Path | None = None
        try:
            with lock(compute_lock_path_in(self._base_dir)):
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
        except LockError:
            if tombstone_path is None:
                raise

        if tombstone_path is None:
            return False
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
    def from_artifact[TSpec: Spec](
        cls: type[TSpec], artifact: ArtifactSpec | Path
    ) -> TSpec:
        from furu.serializer.artifact import _from_artifact

        if not isinstance(artifact, ArtifactSpec):
            with metadata_path_in(artifact).open(encoding="utf-8") as f:
                artifact = ArtifactSpec.model_validate(json.load(f)["artifact"])
        return _from_artifact(artifact, cls)

    @final
    @cached_property
    def _artifact_data(  # TODO: make sure this doesn't prevent garbage collection
        self,
    ) -> dict[str, JsonValue]:
        return _to_json(
            self,
            declared_type=type(self),
            artifact_serializers=self.artifact_serializers,
        )  # ty:ignore[invalid-return-type] # TODO: check this or make _to_json return dict[str, JsonValue] or typed value

    @final
    @cached_property
    def _artifact_data_for_hash(self) -> JsonValue:
        return _to_json(
            self,
            declared_type=type(self),
            artifact_serializers=self.artifact_serializers,
            for_hash=True,
        )

    @final
    @cached_property
    def _artifact_hash(  # TODO: should this be __hash__?
        self,
    ) -> str:
        return _hash_dict_deterministically(self._artifact_data_for_hash)

    @final
    @cached_property
    def _schema_data(
        self,
    ) -> JsonValue:
        return _schema_type(
            type(self),
            set(),
            artifact_serializers=self.artifact_serializers,
        )

    @final
    @cached_property
    def _artifact_schema_hash(self) -> str:
        return _hash_dict_deterministically(
            _schema_type(
                type(self),
                set(),
                artifact_serializers=self.artifact_serializers,
                for_hash=True,
            )
        )

    @final
    @cached_property
    def _fully_qualified_name(self) -> str:
        return fully_qualified_name(type(self))

    @final
    @cached_property
    def _base_dir(self) -> Path:
        return (
            self._storage_root
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
