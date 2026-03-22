import os
import pickle
import secrets
import shutil
import traceback
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from functools import cache, cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    get_type_hints,
)

from pydantic import TypeAdapter, ValidationError

from furu.config import config
from furu.locking import LockLostError, lock

# from furu.locking import run_with_lease_and_pickle_result
from furu.metadata import RunningMetadata
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import (
    JsonValue,
    _hash_dict_deterministically,
    _nfs_safe_unique_name,
    fully_qualified_name,
)

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


class validate:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner: Any, name: str) -> None:
        _ensure_furu_validators(owner)
        owner.__furu_validators__.append(self.fn)
        setattr(owner, name, self.fn)


def _ensure_furu_validators(cls: Any) -> None:
    if "__furu_validators__" in cls.__dict__:
        return

    validators: list = []
    for base in cls.__bases__:
        validators.extend(getattr(base, "__furu_validators__", []))
    cls.__furu_validators__ = validators


@cache
def _type_adapter(tp: Any) -> TypeAdapter[Any]:
    return TypeAdapter(tp)


@cache
def _field_hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls, include_extras=True)


def _validate_field_types(instance: Any) -> None:
    hints = _field_hints(type(instance))
    for field in fields(instance):
        expected_type = hints.get(field.name)
        if expected_type is None:
            continue

        value = getattr(instance, field.name)
        try:
            _type_adapter(expected_type).validate_python(value, strict=True)
        except ValidationError as exc:
            raise TypeError(
                f"{type(instance).__qualname__}.{field.name} failed validation against "
                f"{expected_type!r}: {exc}"
            ) from exc


def _run_furu_validation(instance: Any) -> None:
    _validate_field_types(instance)
    for validator in getattr(type(instance), "__furu_validators__", []):
        validator(instance)


def _find_inherited_post_init(cls: type) -> Any:
    for base in cls.__mro__[1:]:
        post_init = base.__dict__.get("__post_init__")
        if post_init is not None:
            return post_init
    return None


class Furu[T](_FuruDataclassTransform, ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Furu:
            return

        _ensure_furu_validators(cls)

        user_post_init = cls.__dict__.get("__post_init__")
        inherited_post_init = _find_inherited_post_init(cls)
        if user_post_init is not None or inherited_post_init is None:

            def __post_init__(self, *args: Any, **kwds: Any) -> None:
                if inherited_post_init is not None:
                    inherited_post_init(self, *args, **kwds)
                if user_post_init is not None:
                    user_post_init(self, *args, **kwds)
                _run_furu_validation(self)

            cls.__post_init__ = __post_init__

        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)

    def load_or_create(self, use_lock: bool = True) -> T:
        if self._result_path.exists():
            # TODO: validation that its up to date and valid
            with open(self._result_path, "rb") as f:
                return pickle.load(f)

        # TODO: have the logs be per-run with a run id

        self._internal_furu_dir.mkdir(exist_ok=True, parents=True)

        try:
            with (
                lock(self._internal_furu_dir / "compute.lock")
                if use_lock
                else nullcontext()
            ) as has_lock:
                if self._result_path.exists():
                    with open(self._result_path, "rb") as f:
                        return pickle.load(f)

                metadata = RunningMetadata(
                    artifact=self.to_json(),
                    artifact_hash=self.artifact_hash,
                    schema_=self.schema,
                    schema_hash=self.schema_hash,
                    data_path=self.data_dir.resolve(),
                    started_at=datetime.now(timezone.utc),
                )
                self._metadata_path.write_text(metadata.model_dump_json(indent=2))

                result = self._create()

                completed_metadata = metadata.to_complete()

                if has_lock and not has_lock():
                    raise LockLostError(
                        f"lost lock at {self._internal_furu_dir / 'compute.lock'} "
                        "before writing final result"
                    )

                tmp_result_path = self._result_path.with_suffix(".tmp.pkl")
                with tmp_result_path.open("wb") as f:
                    pickle.dump(
                        result,
                        f,
                    )
                    f.flush()  # TODO: Do i need this and the os.fsync?
                    os.fsync(f.fileno())

                if has_lock and not has_lock():
                    raise LockLostError(
                        f"lost lock at {self._internal_furu_dir / 'compute.lock'} "
                        "after writing temporary result"
                    )

                tmp_result_path.rename(self._result_path)
                self._metadata_path.write_text(
                    completed_metadata.model_dump_json(indent=2)
                )

            return result

        except BaseException as exc:
            with (  # TODO: log this to the regular log file
                (
                    self._internal_furu_dir
                    / f"error-{datetime.now():%y%m%d_%H-%M-%S}-{secrets.token_hex(4)}.log"  # TODO: make this part of the regular error
                ).open("a", encoding="utf-8") as f
            ):
                f.write("Traceback (most recent call last):\n")
                f.writelines(
                    traceback.format_list(
                        traceback.extract_stack()[:-1]
                        + traceback.extract_tb(exc.__traceback__)
                    )
                )
                f.writelines(traceback.format_exception_only(type(exc), exc))
                f.write("\n=== Debug Details (with locals) ===\n")
                f.writelines(
                    traceback.TracebackException.from_exception(
                        exc, capture_locals=True
                    ).format(chain=True)
                )
            raise

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
            # TODO: validation that its up to date and valid
            with open(self._result_path, "rb") as f:
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
            with lock(self._internal_furu_dir / "compute.lock"):
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

    @abstractmethod
    def _create(self) -> T:
        raise NotImplementedError("TODO")

    @cache
    def to_json(  # TODO: make sure this doesn't prevent garbage collection
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
        return _hash_dict_deterministically(self.to_json())

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
