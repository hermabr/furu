import enum
import importlib
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel as PydanticBaseModel

from furu.constants import CLASSMARKER, FIELDSMARKER, KINDMARKER
from furu.metadata import ArtifactSpec
from furu.utils import JsonValue, fully_qualified_name

if TYPE_CHECKING:
    from furu.core import Furu

_RESERVED_DICT_KEYS = frozenset({CLASSMARKER, FIELDSMARKER, KINDMARKER})


def to_json(  # TODO: consider caching this (but if i'm going to, I need to figure out how to cache lists and other unhashable objects)
    obj: Any,
) -> JsonValue:
    # TODO: when writing this to metadata, make sure to escape strings etc

    def assert_correct_dict_key(x: Any) -> str:
        if not isinstance(x, str):
            raise ValueError("TODO")
        if x in _RESERVED_DICT_KEYS:
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
        case type():
            return {
                KINDMARKER: "type_ref",
                CLASSMARKER: fully_qualified_name(obj),
            }
        case x if is_dataclass(x):
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(type(x)),
                FIELDSMARKER: {f.name: to_json(getattr(x, f.name)) for f in fields(x)},
            }
        case PydanticBaseModel():
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(type(obj)),
                FIELDSMARKER: {k: to_json(v) for k, v in obj.model_dump().items()},
            }
        case enum.Enum():
            raise NotImplementedError("TODO")  #  return {"__enum__": _type_fqn(obj)}
        case _:
            raise ValueError("unexpected item", obj)  # TODO: explain the error more


def _load_type(qualified_name: str) -> type[Any]:
    module_name, _, class_name = qualified_name.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Expected fully qualified class name, got {qualified_name!r}")

    module = importlib.import_module(module_name)
    value = getattr(module, class_name)
    if not isinstance(value, type):
        raise TypeError(f"{qualified_name!r} resolved to a non-type value")
    return value


def _from_json_field(value: JsonValue, expected_type: Any) -> Any:
    if isinstance(value, dict) and KINDMARKER in value:
        return _from_json(value)

    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if isinstance(value, list):
        item_type = args[0] if args and args[0] is not Ellipsis else Any
        items = [_from_json_field(item, item_type) for item in value]
        return tuple(items) if origin is tuple else items
    if isinstance(value, dict):
        value_type = args[1] if origin is dict and len(args) == 2 else Any
        return {
            key: _from_json_field(child, value_type) for key, child in value.items()
        }
    if expected_type is Path and isinstance(value, str):
        return Path(value)
    return value


def _from_json(value: JsonValue) -> Any:
    match value:
        case list():
            return [_from_json(item) for item in value]
        case dict():
            class_name = value.get(CLASSMARKER)
            field_values = value.get(FIELDSMARKER)
            if value.get(KINDMARKER) == "type_ref" and isinstance(class_name, str):
                return _load_type(class_name)
            if (
                value.get(KINDMARKER) == "instance"
                and isinstance(class_name, str)
                and isinstance(field_values, dict)
            ):
                cls = _load_type(class_name)
                hints = get_type_hints(cls, include_extras=True)
                converted_fields = {
                    name: _from_json_field(field_value, hints.get(name, Any))
                    for name, field_value in field_values.items()
                }
                return cls(**converted_fields)
            return {key: _from_json(child) for key, child in value.items()}
        case _:
            return value


def _from_artifact[T: "Furu"](artifact: ArtifactSpec, expected_type: type[T]) -> T:
    furu_obj = _from_json(artifact.data)
    if not isinstance(furu_obj, expected_type):
        raise TypeError(
            "Artifact described "
            + f"{type(furu_obj).__module__}.{type(furu_obj).__qualname__}, "
            + f"expected {expected_type.__module__}.{expected_type.__qualname__}"
        )
    if artifact.artifact_hash != furu_obj.artifact_hash:
        raise ValueError(
            "Artifact hash did not match loaded object: "
            + f"artifact={artifact.artifact_hash[:5]}, "
            + f"loaded={furu_obj.artifact_hash[:5]}"
        )
    if artifact.schema_hash != furu_obj.artifact_schema_hash:
        raise ValueError(
            "Artifact schema hash did not match loaded object: "
            + f"artifact={artifact.schema_hash[:5]}, "
            + f"loaded={furu_obj.artifact_schema_hash[:5]}"
        )
    return furu_obj
