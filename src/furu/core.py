from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
)

from pydantic import JsonValue

from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import _hash_dict_deterministically

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

    def cached_or_create(self) -> T:
        raise NotImplementedError("TODO")

    def load_if_exists(self) -> T:
        raise NotImplementedError("TODO")

    @abstractmethod
    def _build(self) -> T:
        raise NotImplementedError("TODO")

    @cached_property
    def furu_hash(self) -> str:
        return _hash_dict_deterministically(self.to_json())

    def to_json(self) -> JsonValue:
        return _to_json(self)

    def from_json(self) -> Self:
        raise NotImplementedError("TODO")

    @cached_property
    def furu_schema(self) -> JsonValue:
        return _schema_type(type(self), set())

    @cached_property
    def furu_schema_hash(self) -> str:
        return _hash_dict_deterministically(self.furu_schema)
