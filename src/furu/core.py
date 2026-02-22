from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
)

from pydantic import JsonValue

from furu.config import config
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import _hash_dict_deterministically, fully_qualified_name

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


class Furu[T](_FuruDataclassTransform, ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Furu:
            return
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)

    def load_or_create(self) -> T:
        raise NotImplementedError("TODO")

    def try_load(self) -> T:
        raise NotImplementedError("TODO")

    @abstractmethod
    def _build(self) -> T:
        raise NotImplementedError("TODO")

    @cache
    def to_json(
        self,
    ) -> JsonValue:  # TODO: is there a better name since only the fields are converted to json, not results etc
        # TODO: decide if this should have the furu_ namespace
        return _to_json(self)

    @classmethod
    def from_json(self) -> Self:
        raise NotImplementedError("TODO")

    @cached_property
    def artifact_hash(  # TODO: rename since its confusing that both schema and furu_hash define the dir
        self,
    ) -> str:  # TODO: rename this? i prefer not overriding __hash__ to make it explicit that furu hash is different
        return _hash_dict_deterministically(self.to_json())

    @cached_property
    def schema(
        self,
    ) -> JsonValue:
        return _schema_type(type(self), set())

    @cached_property
    def schema_hash(self) -> str:
        return _hash_dict_deterministically(self.schema)

    @cached_property
    def data_dir(self) -> Path:  # TODO: rename to something like data_dir instead?
        return (
            config.base_directory
            / "data"
            / Path(*fully_qualified_name(type(self)).split("."))
            / self.schema_hash
            / self.artifact_hash
        )

    @cached_property
    def _internal_furu_dir(self) -> Path:
        return self.data_dir / ".furu"
