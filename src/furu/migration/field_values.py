"""Transform stored artifact fields through a migration chain."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

from furu.constants import FIELDSMARKER
from furu.migration.scanner import _ClassResolution
from furu.migration.steps import (
    Added,
    MigrationError,
    MovedFrom,
    Renamed,
    Retyped,
    Rewrite,
    _describe_step,
)
from furu.utils import JsonFields, JsonValue

if TYPE_CHECKING:
    from furu.core import Spec


class _SourceFields(Mapping[str, JsonValue]):
    def __init__(self, fields: JsonFields, description: str) -> None:
        self._fields = fields
        self._description = description

    def __getitem__(self, key: str) -> JsonValue:
        try:
            return self._fields[key]
        except KeyError:
            raise KeyError(
                f"{key!r} is not a source field for {self._description}; "
                f"source fields: {sorted(self._fields)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)


def _added_default_fields(obj: Spec[Any], resolution: _ClassResolution) -> JsonFields:
    names = set(resolution.added_current_name.values())
    if not names:
        return {}
    field_by_name = {field.name: field for field in dataclasses.fields(type(obj))}
    replacements: dict[str, Any] = {}
    for name in names:
        field = field_by_name[name]
        if field.default is not dataclasses.MISSING:
            replacements[name] = field.default
        else:
            factory = field.default_factory
            assert callable(factory)  # phase-one validation guarantees a default
            replacements[name] = factory()
    defaults_obj = dataclasses.replace(obj, **replacements)
    fields_json = cast(JsonFields, defaults_obj._artifact_data[FIELDSMARKER])
    return {name: fields_json[name] for name in names}


def _apply_steps(
    resolution: _ClassResolution,
    start: int,
    source_fields: JsonFields,
    added_defaults: JsonFields,
) -> JsonFields:
    fields = dict(source_fields)
    for index in range(start, len(resolution.steps)):
        step = resolution.steps[index]
        description = _describe_step(step)
        match step:
            case Renamed(field=field, to=to):
                if field not in fields:
                    raise MigrationError(
                        f"{description}: stored result has no field {field!r}; "
                        f"stored fields: {sorted(fields)}"
                    )
                fields[to] = fields.pop(field)
            case Added(field=field):
                fields[field] = added_defaults[resolution.added_current_name[index]]
            case Retyped() | MovedFrom():
                pass
            case Rewrite(transform=transform):
                rewritten = dict(transform(_SourceFields(fields, description)))
                if set(rewritten) != set(fields):
                    raise MigrationError(
                        f"{description} must preserve field names: it returned "
                        f"{sorted(rewritten)} for source fields {sorted(fields)}; "
                        "use Renamed/Added for field renames and additions"
                    )
                fields = rewritten
    return fields
