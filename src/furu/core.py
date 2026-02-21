import enum
import hashlib
import json
import types
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    assert_never,
    get_args,
    get_origin,
    get_type_hints,
)

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
ORIGINMARKER = "|origin"
ARGSMARKER = "|args"


class Furu(_FuruDataclassTransform, Generic[T], ABC):
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
                            if not k.startswith("_")
                        }
                    else:
                        return {k: canonicalize(v) for k, v in sorted(item.items())}
                case str() | bool() | int() | float() | None:
                    return item
                case x:
                    assert_never(x)

        canonical_obj = canonicalize(_to_dict(self))

        return _hash_dict_deterministically(canonical_obj)

    @cached_property
    def furu_schema(self) -> JsonValue:
        seen = set()

        def _schema_dataclass(tp: type) -> JsonValue:
            if tp in seen:
                return {CLASSMARKER: _type_fqn(tp)}
            seen.add(tp)

            hints = get_type_hints(tp, include_extras=True)
            return {
                CLASSMARKER: _type_fqn(tp),
                "fields": {
                    f.name: _schema_type(hints.get(f.name, f.type))
                    for f in sorted(fields(tp), key=lambda f: f.name)
                    if not f.name.startswith("_")
                },
            }

        def _schema_type(tp: Any) -> JsonValue:
            origin = get_origin(tp)

            if (isinstance(tp, type) and is_dataclass(tp)) or (
                is_dataclass(origin) and (tp := origin)
            ):
                return _schema_dataclass(tp)

            if origin in (typing.Union, types.UnionType):
                return sorted(
                    [_schema_type(a) for a in get_args(tp)], key=_stable_json_dump
                )
            elif origin is not None:
                assert (args := get_args(tp))  # TODO: maybe i need to remove this?
                return {
                    ORIGINMARKER: _type_fqn(origin),
                    ARGSMARKER: sorted(
                        [_schema_type(a) for a in args], key=_stable_json_dump
                    ),
                }

            if tp in [str, float, int, bool]:
                return _type_fqn(tp)
            elif isinstance(tp, str):
                return tp
            elif tp in [list, tuple, dict]:
                assert get_args(tp) == ()
                return _type_fqn(tp)
            elif isinstance(tp, typing.TypeVar):
                return repr(tp)
            elif isinstance(tp, enum.EnumType):
                return _type_fqn(tp)
            assert False, f"TODO: unexpected type value {tp=}"

        seen = set()

        return _schema_type(type(self))

    @cached_property
    def furu_schema_hash(self) -> str:
        return _hash_dict_deterministically(self.furu_schema)


def _stable_json_dump(x: JsonValue) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _hash_dict_deterministically(obj: JsonValue) -> str:
    json_str = _stable_json_dump(obj)

    return hashlib.blake2s(
        json_str.encode(),
        digest_size=10,  # TODO: make this digest size configurable and include a script for estimating likelihood of crashing. right now, i think there is a 1e-08 chance of a collision with 155M items with the same schema and namespace
    ).hexdigest()


def _type_fqn(tp: type) -> str:
    mod = tp.__module__
    qualname = tp.__qualname__
    if mod == "__main__":  # TODO: allow overwriting
        raise ValueError("Cannot serialize objects from __main__ module")
    elif "<locals>" in mod:  # TODO: allow overwriting
        raise ValueError("TODO: msg")
    elif "." in qualname:
        raise ValueError("TODO: msg")
    elif isinstance(tp, enum.Enum):
        raise ValueError(
            "TODO: support this in the future"
        )  # return f"{mod}.{qualname}.{obj.name}"
    return f"{mod}.{qualname}"


# TODO: should i cache this?
def _to_dict(obj: Any) -> JsonValue:

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
