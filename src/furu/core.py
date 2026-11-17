from __future__ import annotations

import logging
import shutil
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from inspect import get_annotations
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    final,
)

from furu._batched import _BatchedCreate, _BatchedHook, batched
from furu.dependencies import (
    _Dependency,
    dependency_field,
    is_under_creation,
    record_dependency,
)
from furu.locking import LockError, is_active_lock, lock
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.migration.schema import held
from furu.migration.steps import MigrationStep, naming, stored_name
from furu.provenance import Provenance
from furu.resources import Worker
from furu.result.codec import Codec
from furu.serializer.artifact import hashed_fields
from furu.serializer.artifact import to_json as _to_json
from furu.serializer.registry import Serializer
from furu.spec_metadata import Metadata, Throttle
from furu.storage._layout import (
    attempt_dir_in,
    compute_lock_path_in,
    provenance_path_in,
    spec_path_in,
)
from furu.storage.directory import SpecDirectory
from furu.storage.stored import load_stored_result, stored_version
from furu.utils import (
    JsonValue,
    _hash_dict_deterministically,
    nfs_safe_unique_name,
    object_id_from_parts,
    spec_label,
)

if TYPE_CHECKING:
    from typing import dataclass_transform

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


class Missing(Exception):
    pass


_SPEC_CLASS_ATTRIBUTES = frozenset(
    {"migrations", "throttle", "result_codecs", "artifact_serializers"}
)
_RESERVED_FIELD_NAMES = frozenset(
    {
        "create",
        "metadata",
        "status",
        "directory",
        "load_existing",
        "delete",
        "migrate",
        "migrations",
        "provenance",
        "throttle",
        "result_codecs",
        "artifact_serializers",
        "runs_on",
    }
)


class Spec[T](_FuruDataclassTransform, ABC):
    """One result, described by its fields; ``create()`` computes it once.

    A spec's id is its class, the names of the class's fields and
    ``@furu.dependency`` methods, and a fingerprint of its field values.
    Changing a field therefore gives every spec of the class a new id, and
    leaves the stored results behind. A step in ``migrations`` (``Added``,
    ``Renamed``, ``MovedFrom``) keeps them: the spec is named the way it was
    before the change.
    """

    _furu_create_hook: ClassVar[Callable[..., Any] | _BatchedHook]
    throttle: ClassVar[Throttle | None] = None
    migrations: ClassVar[tuple[MigrationStep, ...]] = ()
    result_codecs: ClassVar[tuple[type[Codec], ...]] = ()
    artifact_serializers: ClassVar[tuple[type[Serializer], ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Spec:
            return

        annotations = get_annotations(cls, eval_str=False)
        dependencies = {
            name: value
            for name, value in cls.__dict__.items()
            if isinstance(value, _Dependency)
        }
        for name, value in dependencies.items():
            setattr(cls, name, dependency_field(cls, name, value))
            annotations[name] = value.func.__annotations__["return"]
        # Assigned, not edited in place: reading cls.__annotations__ on a class
        # without its own can return a dict shared with other classes.
        cls.__annotations__ = annotations
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
                and not isinstance(
                    value, (classmethod, property, cached_property, _BatchedCreate)
                )
            ):
                raise TypeError(f"{cls.__name__}.{name} must have a type annotation")

        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)
        for name, value in dependencies.items():
            # Set after the dataclass has taken the field: the first read
            # calls the method and keeps what it returned.
            lazy = cached_property(value.func)
            lazy.__set_name__(cls, name)
            setattr(cls, name, lazy)
        naming(cls)  # refuses steps that do not fit the fields
        match cls.__dict__.get("create"):
            case None:
                pass
            case batched():
                raise TypeError(
                    f"{cls.__qualname__}.create: @furu.batched needs a batch "
                    "key function: @furu.batched(batch_key)"
                )
            case _BatchedCreate() as hook:
                cls._furu_create_hook = _BatchedHook(hook.func, hook.batch_fn)
                del cls.create  # unshadow the inherited create verb
            case hook:
                cls._furu_create_hook = hook
                del cls.create  # unshadow the inherited create verb

    def create(self) -> T:
        """Return this spec's result, building it first if it's missing.

        Declare the specs a create depends on as fields or ``@furu.dependency`` so
        the executor can plan them. A missing spec known only at runtime restarts
        the create from the top, so keep these few and, where possible, request
        them together in one ``furu.create([...])`` call.
        """
        from furu.execution.load_or_create import _load_or_create

        return _load_or_create(self)

    def metadata(self) -> Metadata:
        return Metadata()

    def runs_on(self, worker: Worker) -> bool:
        """Whether a worker can run this spec, e.g. ``worker.gpus >= 1``."""
        return True

    @final
    @cached_property
    def _metadata(self) -> Metadata:
        return self.metadata()

    @property
    def logger(self) -> logging.Logger:
        return get_logger()

    @final
    @property
    def directory(self) -> SpecDirectory:
        if not is_under_creation(self):
            raise RuntimeError(
                f"{self._log_label}.directory is only available inside this "
                "spec's own create() hook. To use files another spec produced, "
                "include their paths in its result and go through "
                "create()/load_existing() so the dependency is recorded. If "
                "create() spawns threads that need .directory, propagate "
                "context with contextvars.copy_context()."
            )
        return SpecDirectory(attempt_dir_in(self._base_dir))

    @final
    @cached_property
    def object_id(self) -> str:
        return object_id_from_parts(
            fully_qualified_name=self._stored_name,
            class_part=self._class_part,
            artifact_hash=self._artifact_hash,
        )

    @final
    def load_existing(self) -> T:
        if (version := stored_version(self)) is not None:
            record_dependency(self, version)
            return load_stored_result(self, version)
        raise Missing(
            f"{self._log_label}.load_existing() could not find a result. "
            "load_existing() only loads existing results; use create() to compute "
            "missing results."
        )

    @final
    def provenance(self) -> Provenance:
        if (version := stored_version(self)) is None:
            raise Missing(
                f"{self._log_label}.provenance() could not find a result. "
                "Provenance is recorded when a result is computed; use create() "
                "to compute it first."
            )
        record_dependency(self, version)
        path = provenance_path_in(version)
        if not path.exists():
            raise Missing(
                f"{self._log_label}.provenance(): the result exists but "
                f"{path.name} is missing beside it. Every stored result should "
                "have one; this indicates the result directory was tampered "
                "with or written by a furu version without provenance capture."
            )
        return Provenance.model_validate_json(path.read_text(encoding="utf-8"))

    @final
    @property
    def status(self) -> Literal["missing", "running", "incomplete", "done"]:
        if stored_version(self) is not None:
            return "done"
        if is_active_lock(compute_lock_path_in(self._base_dir)):
            return "running"
        if attempt_dir_in(self._base_dir).exists():
            # An attempt started and did not finish (crashed, failed, or
            # blocked on a dependency).
            return "incomplete"
        return "missing"

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
    def from_artifact(cls, artifact: ArtifactSpec | Path) -> Self:
        from furu.serializer.artifact import _from_artifact

        if not isinstance(artifact, ArtifactSpec):
            artifact = ArtifactSpec.model_validate_json(
                spec_path_in(artifact).read_text(encoding="utf-8")
            )
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
    def _artifact_hash(  # TODO: should this be __hash__?
        self,
    ) -> str:
        # One fingerprint per field, so writing a large value apart from the
        # spec that holds it never changes the id.
        return _hash_dict_deterministically(
            {
                name: _hash_dict_deterministically(value)
                for name, value in hashed_fields(
                    self, self.artifact_serializers, held[type(self)]
                ).items()
            }
        )

    @final
    @property
    def _class_part(self) -> str:
        return naming(type(self)).class_part

    @final
    @property
    def _stored_name(self) -> str:
        return stored_name(type(self))

    @final
    @cached_property
    def _base_dir(self) -> Path:
        return (
            self._metadata.storage
            / self._stored_name
            / self._class_part
            / self._artifact_hash
        )

    @final
    @cached_property
    def _log_label(self) -> str:
        return spec_label(self._stored_name, self._class_part, self._artifact_hash)
