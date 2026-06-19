from __future__ import annotations

import typing
from dataclasses import MISSING, Field, dataclass, field as dataclass_field, fields
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel as PydanticBaseModel


@dataclass(frozen=True, slots=True)
class _SkipHashMarker:
    pass


_SKIPHASH_MARKER = _SkipHashMarker()


if TYPE_CHECKING:
    type skiphash[T] = T
else:

    class skiphash:
        def __new__(cls, *args: object, **kwargs: object) -> skiphash:
            raise TypeError("furu.skiphash is a type annotation, not a value")

        def __class_getitem__(cls, item: object) -> object:
            return Annotated[item, _SKIPHASH_MARKER]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    declared_type: object
    identity_type: object
    skiphash: bool
    dataclass_field: Field[Any] | None = None


def _is_skiphash_marker(value: object) -> bool:
    return isinstance(value, _SkipHashMarker)


def split_skiphash(declared_type: object) -> tuple[object, bool]:
    if isinstance(declared_type, typing.TypeAliasType):
        value, skip = split_skiphash(declared_type.__value__)
        if skip:
            return value, True
        return declared_type, False

    if get_origin(declared_type) is Annotated:
        base, *metadata = get_args(declared_type)
        kept_metadata = [
            metadata_item
            for metadata_item in metadata
            if not _is_skiphash_marker(metadata_item)
        ]
        if len(kept_metadata) != len(metadata):
            if kept_metadata:
                return Annotated[base, *kept_metadata], True
            return base, True

    return declared_type, False


def strip_skiphash(declared_type: object) -> object:
    return split_skiphash(declared_type)[0]


def dataclass_field_specs(
    tp: type,
    *,
    include_skiphash: bool,
) -> tuple[FieldSpec, ...]:
    hints = get_type_hints(tp, include_extras=True)
    specs: list[FieldSpec] = []
    for field in fields(tp):
        declared_type = hints.get(field.name, Any)
        identity_type, skip = split_skiphash(declared_type)
        if skip and not include_skiphash:
            continue
        specs.append(
            FieldSpec(
                name=field.name,
                declared_type=declared_type,
                identity_type=identity_type,
                skiphash=skip,
                dataclass_field=field,
            )
        )
    return tuple(specs)


def pydantic_field_specs(
    tp: type[PydanticBaseModel],
    *,
    include_skiphash: bool,
) -> tuple[FieldSpec, ...]:
    hints = get_type_hints(tp, include_extras=True)
    specs: list[FieldSpec] = []
    for name in tp.model_fields:
        declared_type = hints.get(name, Any)
        identity_type, skip = split_skiphash(declared_type)
        if skip and not include_skiphash:
            continue
        specs.append(
            FieldSpec(
                name=name,
                declared_type=declared_type,
                identity_type=identity_type,
                skiphash=skip,
            )
        )
    return tuple(specs)


def prepare_skiphash_dataclass_fields(cls: type) -> None:
    annotations = getattr(cls, "__annotations__", {})
    if not annotations:
        return

    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        return
    for name in annotations:
        _, skip = split_skiphash(hints.get(name, annotations[name]))
        if not skip:
            continue

        default = cls.__dict__.get(name, MISSING)
        if isinstance(default, Field):
            default.compare = False
        elif default is MISSING:
            setattr(cls, name, dataclass_field(compare=False))
        else:
            setattr(cls, name, dataclass_field(default=default, compare=False))
