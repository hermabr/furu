import enum
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, assert_never

from pydantic import BaseModel as PydanticBaseModel
from pydantic import JsonValue

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


T = TypeVar("T")

CLASSMARKER = "|class"


class Furu(_FuruDataclassTransform, Generic[T], ABC):
    def __init_subclass__(cls, **kwargs: object) -> None:
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
        def canonicalize(item: JsonValue) -> JsonValue:
            # TODO: write tests where i try to construct two different objects that are canonicalized the same way
            match item:
                case list():
                    return sorted(canonicalize(x) for x in item)
                case dict():
                    if CLASSMARKER in item:
                        return {
                            k: canonicalize(v)
                            for k, v in item.items()  # TODO: make sure i don't need to sort these since i json dump with sort_keys=True
                            if k == CLASSMARKER or (not k.startswith("_"))
                        }
                    else:
                        return {k: canonicalize(v) for k, v in sorted(item.items())}
                case str() | bool() | int() | float() | None:
                    return item
                case x:
                    assert_never(x)

        canonical_obj = canonicalize(_to_dict(self))

        json_str = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":"))

        return hashlib.blake2s(
            json_str.encode(),
            digest_size=10,  # TODO: make this digest size configurable and include a script for estimating likelihood of crashing. right now, i think there is a 1e-08 chance of a collision with 155M items with the same schema and namespace
        ).hexdigest()

    @cached_property
    def furu_schema_hash(self) -> str:
        print([x.type for x in fields(self)])
        return ""


def _type_fqn(tp: type) -> str:
    mod = tp.__module__
    qualname = tp.__qualname__
    if mod == "__main__":
        raise ValueError("Cannot serialize objects from __main__ module")
    elif "<locals>" in mod:
        raise ValueError("TODO: msg")
    elif "." in qualname:
        raise ValueError("TODO: msg")
    elif isinstance(tp, enum.Enum):
        raise ValueError(
            "TODO: support this in the future"
        )  # return f"{mod}.{qualname}.{obj.name}"
    return f"{mod}.{qualname}"


@cache
def _to_dict(obj: object) -> JsonValue:
    def assert_correct_dict_key(x: object) -> str:
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
        case list() | tuple() | set() | frozenset():
            return [_to_dict(x) for x in obj]
        case dict():
            return {assert_correct_dict_key(k): _to_dict(v) for k, v in obj.items()}
        case x if is_dataclass(x):
            return {
                CLASSMARKER: _type_fqn(type(x)),
                **{f.name: _to_dict(getattr(x, f.name)) for f in fields(x)},
            }
        case PydanticBaseModel():
            return {
                CLASSMARKER: _type_fqn(type(obj)),
                **{k: _to_dict(v) for k, v in obj.model_dump().items()},
            }
        case enum.Enum():
            raise NotImplementedError("TODO")  #  return {"__enum__": _type_fqn(obj)}
        case _:
            raise ValueError("unexpected item", obj)
