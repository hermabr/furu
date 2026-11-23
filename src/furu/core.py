from __future__ import annotations

import logging
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
    Protocol,
    cast,
    final,
)

from furu._batched import _BatchedCreate, _BatchedHook, batched, is_batched
from furu._declared_types import cached_type_hints
from furu.dependencies import (
    Returned,
    _Dependency,
    checked,
    creation,
    dependency_field,
    remembered_methods,
    remembers,
    returned_attributes,
)
from furu.logging import get_logger
from furu.migration.schema import held
from furu.migration.steps import MigrationStep, naming, stored_name
from furu.resources import Worker
from furu.serializer.artifact import field_fingerprints, is_default
from furu.serializer.registry import Serializer
from furu.spec_metadata import Metadata, Throttle
from furu.storage import attempt_lock
from furu.storage._layout import (
    attempt_parent_in,
    run_dir_in,
    run_name,
    version_dirs_in,
)
from furu.storage.directory import SpecDirectory
from furu.storage.stored import Stored, load_rows, stored_row
from furu.utils import _hash_dict_deterministically

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


_SPEC_CLASS_ATTRIBUTES = frozenset({"migrations", "throttle", "artifact_serializers"})
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
        "artifact_serializers",
        "runs_on",
    }
)


class Creates[T](Protocol):
    """Any spec whose ``create()`` gives a ``T``, e.g. ``list[furu.Creates[Path]]``.

    For annotations only: a spec's result type is what its ``create()`` is
    annotated to return, so ``Spec`` itself takes no type parameter.
    """

    @property
    def create(self) -> Callable[[], T]: ...

    def load_existing(self) -> T: ...

    @property
    def done(self) -> bool: ...


class Spec(_FuruDataclassTransform, ABC):
    """One result, described by its fields; ``create()`` computes it once.

    The result's type is the return annotation of ``create()``, for type
    checkers; furu never evaluates it. A result with several parts can be a
    dataclass nested in the spec, named ``Result``. Only classes stored in a
    result may be nested, and only directly in a spec. A value in a result is
    stored with a registered codec, else a built-in one;
    ``Annotated[..., SomeCodec]`` on a field of a result dataclass picks one
    for that field.

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
    artifact_serializers: ClassVar[tuple[type[Serializer], ...]] = ()
    # Whether the spec was built from a plan line, which gives it what naming
    # finds. The script that wrote the line named the spec and checked its class.
    _from_line: ClassVar[bool] = False

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
            lazy = Returned(value.func, value.remember)
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

    def create(self) -> Any:
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
        if self.object_id not in creation().specs:
            raise RuntimeError(
                f"{self.object_id}.directory is only available inside this "
                "spec's own create() hook. To use files another spec produced, "
                "include their paths in its result and go through "
                "create()/load_existing() so the dependency is recorded. If "
                "create() spawns threads that need .directory, propagate "
                "context with contextvars.copy_context()."
            )
        if is_batched(type(self)):
            raise RuntimeError(
                f"{self.object_id}.directory: a batch has one directory, which "
                "its create() gets as the second argument: "
                "create(specs, directory). Name the files you write per spec by "
                "the spec's fields, since the same spec can land in another batch."
            )
        attempt_dir = creation().attempt_dir
        assert attempt_dir is not None  # a create() runs in a working folder
        return SpecDirectory(attempt_dir)

    @final
    @cached_property
    def object_id(self) -> str:
        return f"{self._stored_name}:{self._class_part}:{self._artifact_hash}"

    @final
    def load_existing[T](self: Creates[T]) -> T:
        assert isinstance(self, Spec)
        if (row := stored_row(self)) is not None:
            return cast(T, load_rows([self], [row])[0])
        checked("caught furu.Missing")
        # A spec without the decorator is not looked for in batch folders.
        row = None if is_batched(type(self)) else Stored().listed(self, batches=True)
        raise Missing(
            f"{self.object_id}.load_existing() could not find a result. "
            + (
                f"Its result is in the batch folder {row.run}, stored when its "
                "class had @furu.batched. Put the decorator back to load it."
                if row
                else "load_existing() only loads existing results; use create() "
                "to compute missing results."
            )
        )

    @final
    @property
    def done(self) -> bool:
        """Whether a result is stored and its code still matches.

        A spec that is being computed, or whose attempt stopped, is not done.
        """
        checked("read .done")
        return stored_row(self) is not None

    @final
    def delete(self, mode: Literal["prompt", "force"] = "prompt") -> bool:
        """Remove this spec's runs under every code version, and its working folder.

        A traced class keeps each version's trace.json, so what was built on
        the deleted results stays done. A spec with ``@furu.batched`` is a row
        in a run that other specs share, so this raises and names that run.
        A working folder that is not free to take raises too: its create() may
        be running.
        """
        if is_batched(type(self)):
            row = stored_row(self)
            raise RuntimeError(
                f"{self.object_id}.delete(): a spec with @furu.batched has no "
                "folder of its own. "
                + (
                    "It has no stored result."
                    if row is None
                    else f"Its result is a row in {row.run}, and deleting that "
                    "folder deletes every spec in the batch."
                )
            )
        parent = attempt_parent([self])
        try:
            attempt = attempt_lock.free(parent)
        except attempt_lock.HeldError as exc:
            raise attempt_lock.HeldError(f"{self.object_id}.delete(): {exc}") from None
        runs = [
            run
            for version in version_dirs_in(self._class_dir)
            if (run := run_dir_in(version, self._artifact_hash)).is_dir()
        ]
        folders = [attempt, *runs] if attempt else runs
        if not folders:
            return False
        if mode == "prompt":
            listed = ", ".join(map(str, folders))
            answer = input(f"Do you want to delete {listed}? [y/N] ")
            if answer.strip().lower() != "y":
                return False
        # The working folder first: if a worker took it since, this raises
        # with every published run still there.
        attempt_lock.discard(parent)
        if runs:
            parent.parent.mkdir(exist_ok=True)
            for run in runs:
                # Renamed first, so no reader finds half a run.
                attempt_lock.trash(run, into=parent.parent)
        return True

    @final
    @cached_property
    def _named(self) -> tuple[str, dict[str, str]]:
        """The fingerprint of the field values, and of what each dependency method returned.

        The second is by field, so that the spec's line is written without the
        returned values being written out again. A method that returned its
        declared default is left out of the id, and of this unless the method
        has ``remember=True``: its field may still have to be compared.
        """
        # One fingerprint per field, so writing a large value apart from the
        # spec that holds it never changes the id.
        cls = type(self)
        serializers, nested = self.artifact_serializers, held[cls]
        remembered = remembered_methods(cls) if remembers() else {}
        fingerprints = field_fingerprints(self, serializers, nested, leave=remembered)
        returned = {
            attribute: fingerprints[name]
            for attribute, name, _ in naming(cls).fields
            if attribute in returned_attributes(cls) and name in fingerprints
        }
        for attribute, name, added in naming(cls).fields if remembered else ():
            if (method := remembered.get(attribute)) is not None:
                # Remembered under every fingerprint before it, which cover
                # all that the method can read of the spec.
                found = returned[attribute] = method.remembered(
                    self, name, _hash_dict_deterministically(fingerprints)
                )
                declared = cached_type_hints(cls).get(attribute, Any)
                if added is None or not is_default(
                    added, found, declared, serializers, nested
                ):
                    fingerprints[name] = found
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


def attempt_parent(specs: Sequence[Spec]) -> Path:
    """The folder that holds the working folder of the run with ``specs`` as its rows."""
    return attempt_parent_in(
        specs[0]._class_dir, run_name([spec._artifact_hash for spec in specs])
    )
