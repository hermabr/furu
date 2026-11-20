import enum
import numbers
from collections.abc import Callable, Collection, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, cast, get_args

from pydantic import BaseModel as PydanticBaseModel
from pydantic import JsonValue

from furu import constants
from furu._declared_types import (
    cached_type_hints,
    child_declared_type,
    item_declared_types,
    strip_annotated,
)
from furu.constants import (
    CLASSMARKER,
    FIELDSMARKER,
    KINDMARKER,
    SERIALIZERMARKER,
    VALUEMARKER,
)
from furu.migration.steps import Added, naming, stored_name
from furu.serializer.registry import (
    Serializer,
    SerializerMeta,
    _annotated_serializer,
)
from furu.utils import (
    _fingerprint,
    _stable_json_dump,
    fully_qualified_name,
    resolve_fully_qualified_name,
)

_RESERVED_DICT_KEYS = frozenset(
    {CLASSMARKER, FIELDSMARKER, KINDMARKER, SERIALIZERMARKER, VALUEMARKER}
)
_PLAIN = frozenset({str, int, float, bool, type(None)})

# A field value whose written form, which is ASCII, is longer than this is a
# line of its own in a plan.
OWN_LINE_BYTES = 1024

# The fingerprint of each such list, tuple, dict or set that a launch has
# named, by the identity of the value, the class that holds it and the field,
# with the value so that the identity stays its own. A value that many specs
# share is written out once.
_remembered: ContextVar[dict[tuple[int, type, str], tuple[object, str]] | None] = (
    ContextVar("furu_remembered_values", default=None)
)
_REMEMBERED = frozenset({list, tuple, dict, set, frozenset})


@contextmanager
def remembering() -> Generator[None]:
    """Remember the fingerprints of large values for one launch."""
    token = _remembered.set({})
    try:
        yield
    finally:
        _remembered.reset(token)


def remembered(value: object, cls: type, attribute: str) -> str | None:
    """The fingerprint this launch remembers for a value in a field of ``cls``."""
    if (memo := _remembered.get()) is None:
        return None
    return memo.get((id(value), cls, attribute), (None, None))[1]


@cache
def _declares_serializer(declared_type: object) -> bool:
    """Whether the type, or a type inside it, names a serializer for its values."""
    if _annotated_serializer(declared_type):
        return True
    declared_type = strip_annotated(declared_type)
    return hasattr(declared_type, "__furu_serializer__") or any(
        map(_declares_serializer, get_args(declared_type))
    )


def _serializer_may_apply(
    declared_type: object, artifact_serializers: tuple[type[Serializer], ...]
) -> bool:
    """Whether a serializer could take a plain value of this type, or one inside it."""
    if artifact_serializers or SerializerMeta._auto_registered_serializers:
        return True
    try:
        return _declares_serializer(declared_type)
    except TypeError:  # the type cannot be a cache key: the full lookup decides
        return True


def hashed_fields(
    obj: object,
    artifact_serializers: tuple[type[Serializer], ...],
    nested: set[type],
) -> dict[str, JsonValue]:
    """A dataclass's field values as an id sees them.

    Each is under the name it had before any declared change, and an added
    field is left out while it equals its declared default. So a declared
    change gives the same id as before it.
    """
    hints = cached_type_hints(type(obj))
    hashed: dict[str, JsonValue] = {}
    for attribute, name, added in naming(type(obj)).fields:
        declared_type = hints.get(attribute, Any)
        value = to_json(
            getattr(obj, attribute), declared_type, artifact_serializers, nested
        )
        # Compared as text, so that a NaN equals itself.
        if added is None or _stable_json_dump(value) != _stable_json_dump(
            to_json(added.default, declared_type, artifact_serializers, nested)
        ):
            hashed[name] = value
    return hashed


def fingerprint(
    value: object,
    cls: type,
    attribute: str,
    declared_type: object,
    artifact_serializers: tuple[type[Serializer], ...],
    nested: set[type],
) -> str:
    """The fingerprint of the value of a field of the spec class ``cls``.

    A launch remembers it when the value is large.
    """
    key = (id(value), cls, attribute)
    memo = _remembered.get() if type(value) in _REMEMBERED else None
    if memo is not None and key in memo:
        return memo[key][1]
    written = _stable_json_dump(
        to_json(value, declared_type, artifact_serializers, nested)
    )
    found = _fingerprint(written)
    if memo is not None and len(written) > OWN_LINE_BYTES:
        memo[key] = (value, found)
    return found


def is_default(
    added: Added,
    found: str,
    declared_type: object,
    artifact_serializers: tuple[type[Serializer], ...],
    nested: set[type],
) -> bool:
    """Whether a field's fingerprint is of the default its step declares, which an id leaves out."""
    return found == _fingerprint(
        _stable_json_dump(
            to_json(added.default, declared_type, artifact_serializers, nested)
        )
    )


def field_fingerprints(
    obj: object,
    artifact_serializers: tuple[type[Serializer], ...],
    nested: set[type],
    leave: Collection[str] = (),
) -> dict[str, str]:
    """One fingerprint per field of a spec, named and left out as in `hashed_fields`.

    The fields in ``leave`` are not read.
    """
    cls = type(obj)
    hints = cached_type_hints(cls)
    fingerprints: dict[str, str] = {}
    for attribute, name, added in naming(cls).fields:
        if attribute in leave:
            continue
        declared_type = hints.get(attribute, Any)
        found = fingerprint(
            getattr(obj, attribute),
            cls,
            attribute,
            declared_type,
            artifact_serializers,
            nested,
        )
        if added is None or not is_default(
            added, found, declared_type, artifact_serializers, nested
        ):
            fingerprints[name] = found
    return fingerprints


def to_json(  # TODO: consider caching this (but if i'm going to, I need to figure out how to cache lists and other unhashable objects)
    obj: Any,
    declared_type: object,
    artifact_serializers: tuple[type[Serializer], ...],
    nested: set[type],
) -> JsonValue:
    """`obj` as JSON, the form an id is a hash of and a plan line holds.

    A spec inside `obj` is written as its id, and a dataclass as
    `hashed_fields` names it, with its class added to `nested`.
    """
    # TODO: when writing this to metadata, make sure to escape strings etc

    # Most of what a spec holds is plain values and flat lists of them. JSON
    # already holds those, so they skip the lookup when no serializer could apply.
    tp = type(obj)
    if (tp in _PLAIN or tp is list or tp is tuple) and not _serializer_may_apply(
        declared_type, artifact_serializers
    ):
        # The checks of type(obj) are exact, which ty does not narrow on.
        if tp in _PLAIN:
            return cast(JsonValue, obj)
        if _PLAIN.issuperset(map(type, obj)):
            items = cast(list[JsonValue], list(obj))
            return items if tp is list else {KINDMARKER: "tuple", VALUEMARKER: items}
    if _is_spec(obj):
        from furu.dependencies import fresh

        return {KINDMARKER: "spec", VALUEMARKER: fresh(obj).object_id}

    def assert_correct_dict_key(x: object) -> str:
        if not isinstance(x, str):
            raise TypeError(
                f"Cannot serialize dict key {x!r} of type {type(x).__name__!r}: "
                "artifact dict keys must be strings"
            )
        if x in _RESERVED_DICT_KEYS:
            raise ValueError(
                f"Cannot serialize dict key {x!r}: "
                "it is reserved by furu artifact serialization"
            )
        return x

    if isinstance(obj, type):
        return {
            KINDMARKER: "type_ref",
            CLASSMARKER: fully_qualified_name(obj),
        }

    if serializer := SerializerMeta.serializer_for_dump(
        obj,
        declared_type=declared_type,
        artifact_serializers=artifact_serializers,
    ):
        return {
            KINDMARKER: "custom",
            SERIALIZERMARKER: serializer._serializer_id(),
            VALUEMARKER: serializer.dump(
                obj,
                declared_type=strip_annotated(declared_type),
            ),
        }

    match obj:
        case None:
            return None
        case enum.Enum():
            raise TypeError(
                f"Cannot serialize enum value {obj!r}: use a Literal field instead, "
                'such as mode: Literal["fast", "slow"]'
            )
        case int() | str() | float() | bool() if tp in _PLAIN:
            return obj
        case numbers.Number() | str():
            raise TypeError(
                f"Cannot serialize {obj!r} of type {tp.__name__!r}: "
                "convert it with int(), float() or str() first"
            )
        case Path():
            return {KINDMARKER: "path", VALUEMARKER: str(obj)}
        case datetime():
            return {KINDMARKER: "datetime", VALUEMARKER: obj.isoformat()}
        case list() | tuple():
            items: JsonValue = [
                to_json(
                    x,
                    declared_type=item_type,
                    artifact_serializers=artifact_serializers,
                    nested=nested,
                )
                for x, item_type in zip(obj, item_declared_types(declared_type))
            ]
            if isinstance(obj, list):
                return items
            return {KINDMARKER: "tuple", VALUEMARKER: items}
        case set() | frozenset():
            element_type = child_declared_type(declared_type, 0)
            return {
                KINDMARKER: "set" if isinstance(obj, set) else "frozenset",
                VALUEMARKER: sorted(
                    (
                        to_json(
                            x,
                            declared_type=element_type,
                            artifact_serializers=artifact_serializers,
                            nested=nested,
                        )
                        for x in obj
                    ),
                    key=_stable_json_dump,
                ),
            }
        case dict():
            value_type = child_declared_type(declared_type, None)
            return {
                assert_correct_dict_key(k): to_json(
                    v,
                    declared_type=value_type,
                    artifact_serializers=artifact_serializers,
                    nested=nested,
                )
                for k, v in obj.items()
            }
        case x if is_dataclass(x):
            # Named first: a class whose name is refused must stay off the list.
            name = stored_name(type(x))
            nested.add(type(x))
            return {
                KINDMARKER: "instance",
                CLASSMARKER: name,
                FIELDSMARKER: hashed_fields(x, artifact_serializers, nested),
            }
        case PydanticBaseModel():
            model_cls = type(obj)
            model_fields = model_cls.model_fields
            hints = cached_type_hints(model_cls)
            return {
                KINDMARKER: "instance",
                CLASSMARKER: fully_qualified_name(model_cls),
                FIELDSMARKER: {
                    k: to_json(
                        getattr(obj, k),
                        declared_type=hints.get(k, Any),
                        artifact_serializers=artifact_serializers,
                        nested=nested,
                    )
                    for k in model_fields
                },
            }
        case _:
            raise TypeError(
                f"Cannot serialize value {obj!r} of type {type(obj).__name__!r} "
                "into a furu artifact; register a Serializer for this type"
            )


def _is_spec(obj: object) -> bool:
    from furu.core import Spec

    return isinstance(obj, Spec)


def check_names(cls: type, written: Collection[str]) -> None:
    """Refuse written fields that are not the class's with its declared steps undone."""
    names = naming(cls)
    added = {name for _, name, step in names.fields if step is not None}
    if not set(names.first_names) <= set(written) <= {*names.first_names, *added}:
        raise ValueError(
            f"fields {sorted(written)} were written for {cls.__name__}, whose "
            f"fields with its migrations undone are {sorted(names.first_names)}"
            + (f", with {sorted(added)} added since" if added else "")
        )


def from_json(
    value: JsonValue,
    declared_type: object,
    nested: set[type],
    spec: Callable[[str], object],
    moved: Mapping[str, str],
) -> object:
    """The value `to_json` wrote.

    `spec` gives the spec with an id, and `moved` says where a class written
    under an old name lives now. A dataclass is read with its declared steps
    applied forwards, a field that was left out filled from its step, and its
    class is added to `nested`.
    """

    def read(child: JsonValue, child_type: object) -> object:
        return from_json(child, child_type, nested, spec, moved)

    if isinstance(value, list):
        return [
            read(item, child_declared_type(declared_type, i))
            for i, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value
    match value:
        case dict() if KINDMARKER not in value:
            return {
                key: read(child, child_declared_type(declared_type, key))
                for key, child in value.items()
            }
        case {constants.KINDMARKER: "spec", constants.VALUEMARKER: str(spec_id)}:
            return spec(spec_id)
        case {
            constants.KINDMARKER: "custom",
            constants.SERIALIZERMARKER: str(serializer_id),
            constants.VALUEMARKER: _,
        }:
            return _load_custom_value(serializer_id, value[VALUEMARKER], declared_type)
        case {constants.KINDMARKER: "path", constants.VALUEMARKER: str(path)}:
            return Path(path)
        case {constants.KINDMARKER: "datetime", constants.VALUEMARKER: str(when)}:
            return datetime.fromisoformat(when)
        case {
            constants.KINDMARKER: "tuple" | "set" | "frozenset" as kind,
            constants.VALUEMARKER: list(written),
        }:
            items = [
                read(item, child_declared_type(declared_type, i))
                for i, item in enumerate(written)
            ]
            if kind == "tuple":
                return tuple(items)
            return set(items) if kind == "set" else frozenset(items)
        case {constants.KINDMARKER: "type_ref", constants.CLASSMARKER: str(name)}:
            return _resolve_serialized_type(name)
        case {
            constants.KINDMARKER: "instance",
            constants.CLASSMARKER: str(name),
            constants.FIELDSMARKER: dict(written),
        }:
            cls = _resolve_serialized_type(moved.get(name, name))
            hints = cached_type_hints(cls)
            if not is_dataclass(cls):
                return cls(
                    **{
                        name: read(child, hints.get(name, Any))
                        for name, child in written.items()
                    }
                )
            nested.add(cls)
            check_names(cls, written)
            takes = {f.name for f in fields(cls) if f.init}
            values: dict[str, object] = {}
            for attribute, name, added in naming(cls).fields:
                if attribute not in takes:
                    continue
                if name in written:
                    values[attribute] = read(written[name], hints.get(attribute, Any))
                else:
                    # check_names let it be left out, so a step added it.
                    assert added is not None
                    values[attribute] = added.default
            return cls(**values)
    raise TypeError(f"Cannot read the written value {value!r}")


def _load_custom_value(
    serializer_id: str, written: JsonValue, expected_type: object
) -> object:
    serializer = resolve_fully_qualified_name(serializer_id)
    if not (isinstance(serializer, type) and issubclass(serializer, Serializer)):
        raise TypeError(f"{serializer_id} is not a Serializer")
    return serializer.load(written, declared_type=strip_annotated(expected_type))


def _resolve_serialized_type(class_name: str) -> type:
    value = resolve_fully_qualified_name(class_name)
    if isinstance(value, type):
        return value
    raise TypeError(f"{class_name!r} resolved to a non-type value")
