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
from furu.result import load_result_bundle
from furu.result.codec import ResultRegistry, _default_result_registry
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.storage_layout import (
    compute_lock_path_in,
    internal_furu_dir_in,
    result_link_path_in,
    result_manifest_path_in,
)
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

    from furu.metadata import ArtifactSpec
    from furu.migration import Migration
    from furu.executor import Executor
    from furu.submission import Submission

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
        from furu.execution import _install_create_guards, _resolve_create_mode

        cls._furu_create_mode = _resolve_create_mode(cls)
        _install_create_guards(cls)

    def load_or_create(self, use_lock: bool = True) -> T:
        from furu.execution import load_or_create

        return load_or_create(self, use_lock=use_lock)

    def submit(self, *, executor: Executor) -> Submission[T]:
        from furu.submit import submit

        return submit(self, executor=executor)

    @classmethod
    def from_artifact[TFuru: Furu](cls: type[TFuru], artifact: ArtifactSpec) -> TFuru:
        from furu.serialize import _from_artifact

        return _from_artifact(artifact, cls)

    def status(
        self,
    ) -> Literal[
        "completed", "missing", "running", "failed"
    ]:  # TODO: add queued/waiting state?
        if result_manifest_path_in(self.data_dir).exists():
            return "completed"
        if self.is_migrated():
            return "completed"
        if compute_lock_path_in(self.data_dir).exists():
            return "running"
        if self.data_dir.exists():
            return "failed"
        return "missing"

    def try_load(self) -> T:  # TODO: make a better name for this
        from furu.dependencies import record_dependency_call
        from furu.migration import result_dir_for_loading
        from furu.worker_context import (
            _DependencyNotReady,
            _in_worker_execution_context,
        )

        record_dependency_call(self)
        if (result_dir := result_dir_for_loading(self)) is not None:
            return cast(T, load_result_bundle(result_dir))
        if _in_worker_execution_context():
            raise _DependencyNotReady(
                dependencies=[self],
                call_kind="try_load",
            )
        raise NotImplementedError(
            "TODO: decide if i should throw or return error value"
        )

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        return ()

    def migrate(self) -> bool:
        from furu.migration import migrate

        return migrate(self)

    def is_migrated(self) -> bool:
        return result_link_path_in(self.data_dir).exists()

    def delete(self, mode: Literal["prompt", "force"] = "prompt") -> bool:
        if not self.data_dir.exists():
            return False

        internal_furu_dir_in(self.data_dir).mkdir(exist_ok=True, parents=True)

        tombstone_path: Path | None = None
        try:
            with lock_many([compute_lock_path_in(self.data_dir)]):
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
    def logger(self) -> logging.Logger:
        return get_logger()

    @property
    def result_registry(self) -> ResultRegistry:
        return _default_result_registry()

    def create(self) -> T:
        raise NotImplementedError("TODO")

    @classmethod
    def create_batched[TFuru: Furu](cls: type[TFuru], objs: list[TFuru]) -> list[T]:
        raise NotImplementedError("TODO")

    @cached_property
    def artifact_data(  # TODO: make sure this doesn't prevent garbage collection
        self,
    ) -> dict[str, JsonValue]:
        return _to_json(self)  # ty:ignore[invalid-return-type] # TODO: check this or make _to_json return dict[str, JsonValue] or typed value

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
        return object_id_from_parts(
            fully_qualified_name=self._fully_qualified_name,
            schema_hash=self.artifact_schema_hash,
            artifact_hash=self.artifact_hash,
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
    def _log_label(self) -> str:
        return (
            f"{type(self).__name__}:"
            + f"{self.artifact_schema_hash[:5]}:"
            + f"{self.artifact_hash[:5]}"
        )
