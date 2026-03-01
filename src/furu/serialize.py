import enum
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from furu.constants import CLASSMARKER
from furu.utils import JsonValue, fully_qualified_name


def to_json(  # TODO: consider caching this (but if i'm going to, I need to figure out how to cache lists and other unhashable objects)
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
        case None:
            return None
        case int() | str() | float() | bool():
            return obj
        case Path():
            return str(obj)
        case list() | tuple():
            return [to_json(x) for x in obj]
        case set() | frozenset():
            return sorted([to_json(x) for x in obj], key=repr)
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
