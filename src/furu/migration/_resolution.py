from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from furu.constants import CLASSMARKER, FIELDSMARKER
from furu.migration._steps import (
    Added,
    MigrationError,
    MigrationStep,
    MovedFrom,
    Renamed,
    Retyped,
    Rewrite,
    _describe_step,
)
from furu.serializer.schema import schema_type
from furu.storage._layout import schema_snapshot_path_in_schema_directory
from furu.utils import JsonValue, _stable_json_dump

if TYPE_CHECKING:
    from furu.core import Spec


@dataclass(frozen=True, slots=True)
class _Exact:
    schema: JsonValue


@dataclass(frozen=True, slots=True)
class _Shape:
    shape: JsonValue


class _Anything:
    __slots__ = ()


_ANYTHING = _Anything()
_FieldExpectation: TypeAlias = _Exact | _Shape | _Anything


def _shape_of(schema: JsonValue) -> JsonValue:
    # Erase class internals: a class schema compares by fully qualified name
    # only, so a later change inside e.g. SGD does not refuse to bind.
    if isinstance(schema, dict):
        if CLASSMARKER in schema:
            return schema[CLASSMARKER]
        return {key: _shape_of(value) for key, value in schema.items()}
    if isinstance(schema, list):
        return sorted((_shape_of(item) for item in schema), key=_stable_json_dump)
    return schema


def _shape_of_expectation(expectation: _FieldExpectation) -> JsonValue | None:
    match expectation:
        case _Exact(schema=schema):
            return _shape_of(schema)
        case _Shape(shape=shape):
            return shape
        case _:
            return None


def _expectation_key(expectation: _FieldExpectation) -> JsonValue:
    match expectation:
        case _Exact(schema=schema):
            return ["exact", schema]
        case _Shape(shape=shape):
            return ["shape", shape]
        case _:
            return ["any"]


@dataclass(frozen=True, slots=True)
class _Generation:
    start: int  # steps[start:] lead from this generation to the current schema
    class_name: str
    expectations: Mapping[str, _FieldExpectation]


class _WalkError(Exception):
    def __init__(self, index: int, step: MigrationStep, message: str) -> None:
        super().__init__(message)
        self.index = index
        self.step = step
        self.message = message


def _walk_generations(
    *,
    steps: tuple[MigrationStep, ...],
    class_name: str,
    current_fields: Mapping[str, _FieldExpectation],
    shape_for: Callable[[Retyped], _FieldExpectation],
) -> tuple[dict[int, _Generation], dict[int, str]]:
    expectations: dict[str, _FieldExpectation] = dict(current_fields)
    current_name_of = {name: name for name in expectations}
    added_current_name: dict[int, str] = {}
    generations: dict[int, _Generation] = {}
    class_at = class_name

    def fields_there() -> str:
        return f"fields at that point in the chain: {sorted(expectations)}"

    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        match step:
            case Renamed(field=field, to=to):
                if to not in expectations:
                    raise _WalkError(
                        index, step, f"{to!r} is not a field; {fields_there()}"
                    )
                if field in expectations:
                    raise _WalkError(
                        index,
                        step,
                        f"{field!r} already exists; {fields_there()}",
                    )
                expectations[field] = expectations.pop(to)
                current_name_of[field] = current_name_of.pop(to)
            case Added(field=field):
                if field not in expectations:
                    raise _WalkError(
                        index, step, f"{field!r} is not a field; {fields_there()}"
                    )
                added_current_name[index] = current_name_of.pop(field)
                del expectations[field]
            case Retyped(field=field):
                if field not in expectations:
                    raise _WalkError(
                        index, step, f"{field!r} is not a field; {fields_there()}"
                    )
                expectations[field] = shape_for(step)
            case MovedFrom(fully_qualified_name=name):
                class_at = name
            case Rewrite():
                # A rewrite reshapes values within fields; it preserves the
                # field-name set, so everything else about earlier schemas is
                # unknown but the names still walk.
                expectations = dict.fromkeys(expectations, _ANYTHING)
        generations[index] = _Generation(
            start=index,
            class_name=class_at,
            expectations=dict(expectations),
        )
    return generations, added_current_name


def validate_migration_declaration(cls: type[Spec[Any]]) -> None:
    """Phase one: structural validation at class creation, zero storage access."""
    steps = cls.migrations
    if not isinstance(steps, tuple) or not all(
        isinstance(step, (Renamed, Added, MovedFrom, Retyped, Rewrite))
        for step in steps
    ):
        raise TypeError(
            f"{cls.__name__}.migrations must be a tuple of "
            "Renamed/Added/MovedFrom/Retyped/Rewrite steps"
        )

    fields_by_name = {field.name: field for field in dataclasses.fields(cls)}
    try:
        _, added_current_name = _walk_generations(
            steps=steps,
            class_name="",
            current_fields={name: _ANYTHING for name in fields_by_name},
            shape_for=lambda step: _ANYTHING,
        )
    except _WalkError as error:
        raise TypeError(
            f"{cls.__name__}.migrations[{error.index}] "
            f"({_describe_step(error.step)}): {error.message}"
        ) from None

    for index, current_name in added_current_name.items():
        field = fields_by_name[current_name]
        if (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        ):
            raise TypeError(
                f"{cls.__name__}.migrations[{index}] "
                f"({_describe_step(steps[index])}): field {current_name!r} has no "
                "default; Added can only backfill a field with a default value"
            )


@dataclass(frozen=True, slots=True)
class _GenerationDirectory:
    schema_directory: Path
    snapshot_path: Path
    generation: _Generation | None  # None: orphaned, no chain to current


@dataclass(frozen=True, slots=True)
class _ClassResolution:
    steps: tuple[MigrationStep, ...]
    generations: tuple[_Generation, ...]
    added_current_name: Mapping[int, str]
    directories: tuple[_GenerationDirectory, ...]


_RESOLUTION_CACHE: dict[tuple[type, Path], _ClassResolution | MigrationError] = {}


def _list_schema_directories(tree_directory: Path) -> list[Path]:
    if not tree_directory.exists():
        return []
    return sorted(path for path in tree_directory.iterdir() if path.is_dir())


def _binds(generation: _Generation, snapshot: JsonValue) -> bool:
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get(CLASSMARKER) != generation.class_name:
        return False
    snapshot_fields = snapshot.get(FIELDSMARKER)
    if not isinstance(snapshot_fields, dict):
        return False
    if set(snapshot_fields) != set(generation.expectations):
        return False
    for name, expectation in generation.expectations.items():
        match expectation:
            case _Exact(schema=schema):
                if snapshot_fields[name] != schema:
                    return False
            case _Shape(shape=shape):
                if _shape_of(snapshot_fields[name]) != shape:
                    return False
            case _:
                pass
    return True


def _resolve_class(obj: Spec[Any]) -> _ClassResolution:
    cls = type(obj)
    steps = cls.migrations
    current_schema = cast("dict[str, JsonValue]", obj._schema_data)
    current_class = cast(str, current_schema[CLASSMARKER])
    current_field_schemas = cast("dict[str, JsonValue]", current_schema[FIELDSMARKER])
    current_fields: dict[str, _FieldExpectation] = {
        name: _Exact(schema) for name, schema in current_field_schemas.items()
    }

    def shape_for(step: Retyped) -> _FieldExpectation:
        return _Shape(
            _shape_of(
                schema_type(
                    step.was, set(), artifact_serializers=obj.artifact_serializers
                )
            )
        )

    try:
        generations_by_start, added_current_name = _walk_generations(
            steps=steps,
            class_name=current_class,
            current_fields=current_fields,
            shape_for=shape_for,
        )
    except _WalkError as error:
        raise MigrationError(
            f"{cls.__name__}.migrations[{error.index}] "
            f"({_describe_step(error.step)}): {error.message}"
        ) from None
    generations = tuple(
        generations_by_start[index] for index in sorted(generations_by_start)
    )

    described: dict[str, int] = {}
    for generation in generations:
        key = _stable_json_dump(
            {
                "class": generation.class_name,
                "fields": {
                    name: _expectation_key(expectation)
                    for name, expectation in generation.expectations.items()
                },
            }
        )
        if key in described:
            raise MigrationError(
                f"{cls.__name__}.migrations is ambiguous: the chains starting at "
                f"steps[{described[key]}] and steps[{generation.start}] describe "
                "the same source schema; every source schema must have exactly "
                "one chain to the current schema"
            )
        described[key] = generation.start

    for index, step in enumerate(steps):
        if not isinstance(step, Retyped):
            continue
        post_side = (
            generations_by_start[index + 1].expectations
            if index + 1 < len(steps)
            else current_fields
        )
        post_shape = _shape_of_expectation(post_side[step.field])
        was_shape = _shape_of_expectation(shape_for(step))
        if post_shape is not None and post_shape == was_shape:
            raise MigrationError(
                f"{cls.__name__}.migrations[{index}] ({_describe_step(step)}) is a "
                f"dead step: {step.field!r} already has that type, so the step can "
                "never migrate anything. If old stored values need reshaping - "
                "say a field's class itself changed - that is Rewrite's job."
            )

    trees = {current_class} | {generation.class_name for generation in generations}
    directories: list[_GenerationDirectory] = []
    for tree_name in sorted(trees):
        tree_directory = obj._storage_root / Path(*tree_name.split("."))
        for schema_directory in _list_schema_directories(tree_directory):
            if (
                tree_name == current_class
                and schema_directory.name == obj._artifact_schema_hash
            ):
                continue
            snapshot_path = schema_snapshot_path_in_schema_directory(schema_directory)
            if not snapshot_path.exists():
                continue
            snapshot = cast(
                JsonValue, json.loads(snapshot_path.read_text(encoding="utf-8"))
            )
            matches = [
                generation
                for generation in generations
                if generation.class_name == tree_name and _binds(generation, snapshot)
            ]
            if len(matches) > 1:
                raise MigrationError(
                    f"{cls.__name__}.migrations is ambiguous: the recorded schema "
                    f"at {schema_directory} matches the chains starting at "
                    f"steps[{matches[0].start}] and steps[{matches[1].start}]; "
                    "every source schema must have exactly one chain to the "
                    "current schema"
                )
            directories.append(
                _GenerationDirectory(
                    schema_directory=schema_directory,
                    snapshot_path=snapshot_path,
                    generation=matches[0] if matches else None,
                )
            )
    return _ClassResolution(
        steps=steps,
        generations=generations,
        added_current_name=added_current_name,
        directories=tuple(directories),
    )


def _class_resolution(obj: Spec[Any]) -> _ClassResolution:
    key = (type(obj), obj._storage_root)
    cached = _RESOLUTION_CACHE.get(key)
    if cached is None:
        try:
            cached = _resolve_class(obj)
        except MigrationError as error:
            cached = error
        _RESOLUTION_CACHE[key] = cached
    if isinstance(cached, MigrationError):
        raise cached
    return cached
