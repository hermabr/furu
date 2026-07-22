from __future__ import annotations

import functools
import json
import logging
import shutil
from abc import ABC
from collections.abc import Callable, Hashable
from dataclasses import dataclass, replace
from functools import cached_property
from inspect import get_annotations
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    cast,
    final,
    overload,
)

from furu._declared_types import declared_result_type
from furu.config import get_config
from furu.explain import ExplainDepth
from furu.explain import explain as _explain
from furu.locking import LockError, is_active_lock, lock
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.migration.links import _read_source, result_dir_for_loading
from furu.migration.resolution import validate_embedded_migration_declarations
from furu.migration.stale import raise_if_stale, sideways_status
from furu.migration.steps import MigrationStep, validate_migration_declaration
from furu.provenance import Provenance
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
    provenance_path_in,
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


_SPEC_CLASS_ATTRIBUTES = frozenset(
    {"migrations", "throttle", "result_codecs", "artifact_serializers"}
)
_RESERVED_FIELD_NAMES = frozenset(
    {
        "create",
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


class batched:
    """Marks a create hook as batched: ``@furu.batched(batch_key)``.

    ``batch_key(self)`` returns (what may batch together, max specs per call).
    The decorated hook takes the whole batch (``objs: list[Self]``) and returns
    one result per member, in order; every call receives specs that share one
    batch key, at most the cap of them — locally and on workers, where the
    execution coordinator assembles same-key ready specs into one lease.
    furu captures the hook at class
    creation and installs the create verb in its place: instance access loads
    or creates one result, class access loads or creates a same-class batch.

    ``batch_key`` is typed loosely (``self: Any``) so the hook's inference
    stays exact; define it as a method so its body is checked against Self.
    """

    def __init__(self, batch_fn: Callable[[Any], tuple[Hashable, int]], /) -> None:
        if not callable(batch_fn):
            raise TypeError(
                "@furu.batched needs a batch key function: @furu.batched(batch_key)"
            )
        self.batch_fn = batch_fn

    def __call__[S, T](
        self, func: Callable[[list[S]], list[T]], /
    ) -> _BatchedCreate[S, T]:
        return _BatchedCreate(func, self.batch_fn)


class _BatchedCreate[S, T]:
    def __init__(
        self,
        func: Callable[[list[S]], list[T]],
        batch_fn: Callable[[Any], tuple[Hashable, int]],
        /,
    ) -> None:
        self.func = func
        self.batch_fn = batch_fn

    # The class-access overload must come first so ty resolves
    # ``MyBatched.create([...])`` against it.
    @overload
    def __get__(self, obj: None, objtype: type, /) -> Callable[[list[S]], list[T]]: ...
    @overload
    def __get__(self, obj: S, objtype: type, /) -> Callable[[], T]: ...
    def __get__(self, obj: Any, objtype: type | None = None, /) -> Any:
        raise TypeError(
            "@furu.batched hooks are captured by furu at class creation; "
            "define them on a furu.Spec subclass"
        )


class Spec[T](_FuruDataclassTransform, ABC):
    """Authors implement one create hook, single or batched:

    - ``def create(self) -> T`` computes one result at a time.
    - ``@furu.batched(batch_key)`` over ``def create(objs: list[Self]) ->
      list[T]`` computes a batch at a time, grouped by ``batch_key``.

    furu captures the hook at class creation and installs the create verb
    (load-or-create) in its place.
    """

    # The author's create hook, captured by _install_create_verb: a plain
    # ``(self) -> T`` method, or a staticmethod ``(list[Self]) -> list[T]``
    # when _furu_batch_fn is set (None marks a single-flavored hook).
    _furu_create_hook: ClassVar[Any]
    _furu_batch_fn: ClassVar[Any]
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
                and not isinstance(
                    value, (classmethod, property, cached_property, _BatchedCreate)
                )
            ):
                raise TypeError(f"{cls.__name__}.{name} must have a type annotation")

        validate_cls(cls)
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)
        if cls.migrations:
            validate_migration_declaration(cls)
        validate_embedded_migration_declarations(cls)
        _install_create_verb(cls)

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
                    data_dir=data_dir_in(result_dir.parent),
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
    def provenance(self) -> Provenance:
        result_dir = result_dir_for_loading(self)
        if result_dir is None:
            raise Missing(
                f"{self._log_label}.provenance() could not find a result. "
                "Provenance is recorded when a result is computed; use create() "
                "to compute it first."
            )
        path = provenance_path_in(result_dir.parent)
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
    def status(self) -> Literal["missing", "running", "failed", "done", "stale"]:
        if result_manifest_path_in(self._base_dir).exists():
            return "done"
        has_result_link = result_link_path_in(self._base_dir).exists()
        if has_result_link and _read_source(self._base_dir):
            return "done"
        if is_active_lock(compute_lock_path_in(self._base_dir)):
            return "running"
        if self._base_dir.exists() and not has_result_link:
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


class _BatchedCreateVerb:
    def __get__(self, obj: Spec[Any] | None, objtype: type | None = None) -> Any:
        from furu.execution.load_or_create import _load_or_create

        if obj is None:
            return _load_or_create
        return functools.partial(_load_or_create, obj)


def _install_create_verb(cls: type[Spec[Any]]) -> None:
    """Capture the author's create hook and install the create verb over it."""
    if "create" not in cls.__dict__:
        return
    hook = cls.__dict__["create"]
    if isinstance(hook, batched):
        raise TypeError(
            f"{cls.__qualname__}.create: @furu.batched needs a batch key "
            "function: @furu.batched(batch_key)"
        )
    if isinstance(hook, _BatchedCreate):
        setattr(cls, "_furu_create_hook", staticmethod(hook.func))
        setattr(cls, "_furu_batch_fn", staticmethod(hook.batch_fn))
        setattr(cls, "create", _BatchedCreateVerb())
    else:
        setattr(cls, "_furu_create_hook", hook)
        setattr(cls, "_furu_batch_fn", None)

        @functools.wraps(hook)
        def create_verb(self: Spec[Any]) -> Any:
            from furu.execution.load_or_create import _load_or_create

            return _load_or_create(self)

        setattr(cls, "create", create_verb)
