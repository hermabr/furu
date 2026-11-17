from __future__ import annotations

import logging
import shutil
from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from inspect import get_annotations
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    final,
)

from furu._batched import _BatchedCreate, _BatchedHook, batched, is_batched
from furu.dependencies import (
    Returned,
    _Dependency,
    dependency_field,
    is_under_creation,
    returned_attributes,
)
from furu.locking import lock
from furu.logging import get_logger
from furu.migration.schema import held
from furu.migration.steps import MigrationStep, naming, stored_name
from furu.resources import Worker
from furu.result.codec import Codec
from furu.serializer.artifact import field_fingerprints
from furu.serializer.registry import Serializer
from furu.spec_metadata import Metadata, Throttle
from furu.storage._layout import (
    attempt_dir_in,
    attempts_dir_in,
    lock_path_in,
    run_dir_in,
    run_name,
    version_dirs_in,
)
from furu.storage.directory import SpecDirectory
from furu.storage.stored import load_rows, stored_row
from furu.utils import (
    _hash_dict_deterministically,
    nfs_safe_unique_name,
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
        "done",
        "directory",
        "load_existing",
        "delete",
        "migrations",
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
    # The plan line the spec was built from, which gives it what naming finds.
    # The script that wrote the line named the spec and checked its class.
    _line: ClassVar[dict[str, Any] | None] = None

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
            lazy = Returned(value.func)
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
        if is_batched(type(self)):
            raise RuntimeError(
                f"{self._log_label}.directory: a batch has one directory, which "
                "its create() gets as the second argument: "
                "create(specs, directory). Name the files you write per spec by "
                "the spec's fields, since the same spec can land in another batch."
            )
        return SpecDirectory(self._attempt_dir)

    @final
    @cached_property
    def object_id(self) -> str:
        return f"{self._stored_name}:{self._class_part}:{self._artifact_hash}"

    @final
    def load_existing(self) -> T:
        if (row := stored_row(self)) is not None:
            return load_rows([self], [row])[0]
        raise Missing(
            f"{self._log_label}.load_existing() could not find a result. "
            "load_existing() only loads existing results; use create() to compute "
            "missing results."
        )

    @final
    @property
    def done(self) -> bool:
        """Whether a result is stored and its code still matches.

        A spec that is being computed, or whose attempt stopped, is not done.
        """
        return stored_row(self) is not None

    @final
    def delete(self, mode: Literal["prompt", "force"] = "prompt") -> bool:
        """Remove this spec's runs under every code version, and its working folder.

        A traced class keeps each version's trace.json, so what was built on
        the deleted results stays done. A spec with ``@furu.batched`` is a row
        in a run that other specs share, so this raises and names that run.
        """
        if is_batched(type(self)):
            row = stored_row(self)
            raise RuntimeError(
                f"{self._log_label}.delete(): a spec with @furu.batched has no "
                "folder of its own. "
                + (
                    "It has no stored result."
                    if row is None
                    else f"Its result is a row in {row.run}, and deleting that "
                    "folder deletes every spec in the batch."
                )
            )
        if not self._class_dir.is_dir():
            return False  # nothing of this class was ever stored here
        attempts_dir_in(self._class_dir).mkdir(exist_ok=True)
        with lock(lock_path_in(self._class_dir, self._artifact_hash)):
            folders = [self._attempt_dir] + [
                run_dir_in(version, self._artifact_hash)
                for version in version_dirs_in(self._class_dir)
            ]
            folders = [folder for folder in folders if folder.is_dir()]
            if not folders:
                return False
            if mode == "prompt":
                listed = ", ".join(map(str, folders))
                answer = input(f"Do you want to delete {listed}? [y/N] ")
                if answer.strip().lower() != "y":
                    return False
            for folder in folders:
                # Renamed first, so no reader finds half a run.
                tombstone = nfs_safe_unique_name(folder, name="deleting")
                folder.rename(tombstone)
                shutil.rmtree(tombstone)
        return True

    @final
    @cached_property
    def _named(self) -> tuple[str, dict[str, str]]:
        """The fingerprint of the field values, and of what each dependency method returned.

        The second is by field, so that the spec's line is written without the
        returned values being written out again. A method that returned its
        declared default is left out of the id, and of this.
        """
        # One fingerprint per field, so writing a large value apart from the
        # spec that holds it never changes the id.
        cls = type(self)
        fingerprints = field_fingerprints(self, self.artifact_serializers, held[cls])
        returned = {
            attribute: fingerprints[name]
            for attribute, name, _ in naming(cls).fields
            if attribute in returned_attributes(cls) and name in fingerprints
        }
        return _hash_dict_deterministically(fingerprints), returned

    @final
    @property
    def _artifact_hash(self) -> str:  # TODO: should this be __hash__?
        return self._named[0]

    @final
    @property
    def _returned(self) -> dict[str, str]:
        return self._named[1]

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
    def _class_dir(self) -> Path:
        return self._metadata.storage / self._stored_name / self._class_part

    @final
    @property
    def _attempt_dir(self) -> Path:
        return working_folder([self])

    @final
    @cached_property
    def _log_label(self) -> str:
        return spec_label(self._stored_name, self._class_part, self._artifact_hash)


def working_folder(specs: Sequence[Spec]) -> Path:
    """Where the run that holds ``specs`` as its rows does its work."""
    return attempt_dir_in(
        specs[0]._class_dir, run_name([spec._artifact_hash for spec in specs])
    )
