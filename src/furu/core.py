from __future__ import annotations

import json
import logging
import shutil
from abc import ABC
from dataclasses import dataclass, replace
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
from furu.explain import ExplainDepth
from furu.explain import explain as _explain
from furu.locking import LockError, is_active_lock, lock
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.migration.links import result_dir_for_loading
from furu.migration.stale import raise_if_stale, sideways_status
from furu.migration.steps import MigrationStep, validate_migration_declaration
from furu.result.bundle import load_result_bundle
from furu.result.codec import Codec
from furu.serializer.artifact import to_json as _to_json
from furu.serializer.registry import Serializer
from furu.serializer.schema import schema_type as _schema_type
from furu.spec_metadata import Metadata, Throttle
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

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


class Missing(Exception):
    pass


SpecCreateMode: TypeAlias = Literal["single", "batched"] | None
_SPEC_CLASS_ATTRIBUTES = frozenset(
    {"migrations", "throttle", "result_codecs", "artifact_serializers"}
)
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
        "throttle",
        "result_codecs",
        "artifact_serializers",
    }
)


class Spec[T](_FuruDataclassTransform, ABC):
    _furu_create_mode: ClassVar[SpecCreateMode]
    throttle: ClassVar[Throttle | None] = None
    migrations: ClassVar[tuple[MigrationStep, ...]] = ()
    result_codecs: ClassVar[tuple[type[Codec], ...]] = ()
    artifact_serializers: ClassVar[tuple[type[Serializer], ...]] = ()

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
        for name, value in cls.__dict__.items():
            if (
                not (name.startswith("__") and name.endswith("__"))
                and name not in _SPEC_CLASS_ATTRIBUTES
                and name not in annotations
                and not callable(value)
                and not isinstance(value, (classmethod, property, cached_property))
            ):
                raise TypeError(f"{cls.__name__}.{name} must have a type annotation")

        validate_cls(cls)
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)
        if cls.migrations:
            validate_migration_declaration(cls)
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

    def metadata(self) -> Metadata:
        return Metadata()

    @final
    @cached_property
    def _metadata(self) -> Metadata:
        metadata = self.metadata()
        config = get_config()
        if config.debug_mode:
            return replace(metadata, storage=config.run_directories.objects)
        return metadata

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
        raise_if_stale(self)
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
    def status(self) -> Literal["missing", "running", "failed", "done", "stale"]:
        if result_manifest_path_in(self._base_dir).exists():
            return "done"
        if result_link_path_in(self._base_dir).exists():
            return "done"
        if is_active_lock(compute_lock_path_in(self._base_dir)):
            return "running"
        if self._base_dir.exists():
            return "failed"
        return sideways_status(self)

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
            self._metadata.storage
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
