from __future__ import annotations

import typing
from dataclasses import MISSING, Field, dataclass, field as dataclass_field, fields
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel as PydanticBaseModel


@dataclass(frozen=True, slots=True)
class _SkipHashMarker:
    pass


skip_hash = _SkipHashMarker()


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    declared_type: object
    identity_type: object
    skip_hash: bool
    dataclass_field: Field[Any] | None = None


def _is_skip_hash_marker(value: object) -> bool:
    return isinstance(value, _SkipHashMarker)


def split_skip_hash(declared_type: object) -> tuple[object, bool]:
    if isinstance(declared_type, typing.TypeAliasType):
        value, skip = split_skip_hash(declared_type.__value__)
        if skip:
            return value, True
        return declared_type, False

    if get_origin(declared_type) is Annotated:
        base, *metadata = get_args(declared_type)
        kept_metadata = [
            metadata_item
            for metadata_item in metadata
            if not _is_skip_hash_marker(metadata_item)
        ]
        if len(kept_metadata) != len(metadata):
            if kept_metadata:
                return Annotated[base, *kept_metadata], True
            return base, True

    return declared_type, False


def strip_skip_hash(declared_type: object) -> object:
    return split_skip_hash(declared_type)[0]


def dataclass_field_specs(
    tp: type,
    *,
    include_skip_hash: bool,
) -> tuple[FieldSpec, ...]:
    hints = get_type_hints(tp, include_extras=True)
    specs: list[FieldSpec] = []
    for field in fields(tp):
        declared_type = hints.get(field.name, Any)
        identity_type, skip = split_skip_hash(declared_type)
        if skip and not include_skip_hash:
            continue
        specs.append(
            FieldSpec(
                name=field.name,
                declared_type=declared_type,
                identity_type=identity_type,
                skip_hash=skip,
                dataclass_field=field,
            )
        )
    return tuple(specs)


def pydantic_field_specs(
    tp: type[PydanticBaseModel],
    *,
    include_skip_hash: bool,
) -> tuple[FieldSpec, ...]:
    hints = get_type_hints(tp, include_extras=True)
    specs: list[FieldSpec] = []
    for name in tp.model_fields:
        declared_type = hints.get(name, Any)
        identity_type, skip = split_skip_hash(declared_type)
        if skip and not include_skip_hash:
            continue
        specs.append(
            FieldSpec(
                name=name,
                declared_type=declared_type,
                identity_type=identity_type,
                skip_hash=skip,
            )
        )
    return tuple(specs)


def prepare_skip_hash_dataclass_fields(cls: type) -> None:
    annotations = getattr(cls, "__annotations__", {})
    if not annotations:
        return

    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        return
    for name in annotations:
        _, skip = split_skip_hash(hints.get(name, annotations[name]))
        if not skip:
            continue

        default = cls.__dict__.get(name, MISSING)
        if isinstance(default, Field):
            default.compare = False
        elif default is MISSING:
            setattr(cls, name, dataclass_field(compare=False))
        else:
            setattr(cls, name, dataclass_field(default=default, compare=False))
