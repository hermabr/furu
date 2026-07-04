from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias

from furu.constants import CLASSMARKER
from furu.migration.steps import (
    Added,
    MigrationStep,
    MovedFrom,
    Renamed,
    Retyped,
    Rewrite,
    _describe_step,
)
from furu.utils import JsonValue, _stable_json_dump

_FieldExpectation: TypeAlias = tuple[Literal["exact", "shape", "any"], JsonValue]
_ANY: _FieldExpectation = ("any", None)


def _shape_of(schema: JsonValue) -> JsonValue:
    # Compare class schemas by identity, not by their internal field layout.
    if isinstance(schema, dict):
        if CLASSMARKER in schema:
            return schema[CLASSMARKER]
        return {key: _shape_of(value) for key, value in schema.items()}
    if isinstance(schema, list):
        return sorted((_shape_of(item) for item in schema), key=_stable_json_dump)
    return schema


def _shape_of_expectation(expectation: _FieldExpectation) -> JsonValue | None:
    kind, value = expectation
    if kind == "any":
        return None
    return _shape_of(value) if kind == "exact" else value


@dataclasses.dataclass(frozen=True, slots=True)
class _Generation:
    start: int  # steps[start:] lead from this generation to the current schema
    class_name: str
    expectations: Mapping[str, _FieldExpectation]


def _walk_generations(
    *,
    owner: str,
    error_type: type[Exception],
    steps: tuple[MigrationStep, ...],
    class_name: str,
    current_fields: Mapping[str, _FieldExpectation],
    shape_for: Callable[[Retyped], _FieldExpectation],
) -> tuple[list[_Generation], dict[int, str]]:
    expectations = dict(current_fields)
    current_name_of = {name: name for name in expectations}
    added_current_name: dict[int, str] = {}
    generations: list[_Generation] = []
    class_at = class_name

    def fail(index: int, message: str) -> Exception:
        return error_type(
            f"{owner}.migrations[{index}] ({_describe_step(steps[index])}): "
            f"{message}; fields at that point in the chain: {sorted(expectations)}"
        )

    for index in reversed(range(len(steps))):
        step = steps[index]
        match step:
            case Renamed(field=field, to=to):
                if to not in expectations:
                    raise fail(index, f"{to!r} is not a field")
                if field in expectations:
                    raise fail(index, f"{field!r} already exists")
                expectations[field] = expectations.pop(to)
                current_name_of[field] = current_name_of.pop(to)
            case Added(field=field):
                if field not in expectations:
                    raise fail(index, f"{field!r} is not a field")
                added_current_name[index] = current_name_of.pop(field)
                del expectations[field]
            case Retyped(field=field):
                if field not in expectations:
                    raise fail(index, f"{field!r} is not a field")
                expectations[field] = shape_for(step)
            case MovedFrom(fully_qualified_name=name):
                class_at = name
            case Rewrite():
                expectations = dict.fromkeys(expectations, _ANY)
        generations.append(
            _Generation(
                start=index, class_name=class_at, expectations=dict(expectations)
            )
        )
    generations.reverse()
    return generations, added_current_name
