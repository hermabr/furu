import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic import BaseModel as PydanticBaseModel
from pydantic import JsonValue

from furu.constants import CLASSMARKER
from furu.schema import _schema_type
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

    def get(self) -> T:
        raise NotImplementedError("TODO")

    @abstractmethod
    def _create(self) -> T:
        raise NotImplementedError("TODO")

    @abstractmethod
    def _load(self) -> T:
        raise NotImplementedError("TODO")

    @cached_property
    def furu_hash(self) -> str:
        return _hash_dict_deterministically(to_json(self))

    @cached_property
    def furu_schema(self) -> JsonValue:
        return _schema_type(type(self), set())

    @cached_property
    def furu_schema_hash(self) -> str:
        return _hash_dict_deterministically(self.furu_schema)


@cache
def to_json(
    obj: Any,
) -> JsonValue:
    # TODO: when writing this to metadata, make sure to escape strings etc

    def assert_correct_dict_key(x: Any) -> str:
        if not isinstance(x, str):
            raise ValueError("TODO")
        if x == CLASSMARKER:
            raise ValueError("TODO: write error msg")
        return x

    match obj:
        case int() | str() | float() | bool():
            return obj
        case Path():
            return str(obj)
        case list() | tuple():
            return [to_json(x) for x in obj]
        case set() | frozenset():
            return sorted([to_json(x) for x in obj])  # TODO: will this work?
        case dict():
            return {assert_correct_dict_key(k): to_json(v) for k, v in obj.items()}
        case x if is_dataclass(x):
            return {
                CLASSMARKER: fully_qualified_name(type(x)),
                **{f.name: to_json(getattr(x, f.name)) for f in fields(x)},
            }
        case PydanticBaseModel():
            return {
                CLASSMARKER: fully_qualified_name(type(obj)),
                **{k: to_json(v) for k, v in obj.model_dump().items()},
            }
        case enum.Enum():
            raise NotImplementedError("TODO")  #  return {"__enum__": _type_fqn(obj)}
        case _:
            raise ValueError("unexpected item", obj)  # TODO: explain the error more
