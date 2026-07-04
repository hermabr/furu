from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from furu.constants import CLASSMARKER, FIELDSMARKER
from furu.migration.steps import (
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

# What a snapshot's field schema must look like to bind: "exact" compares the
# recorded schema verbatim, "shape" compares after class-internal erasure
# (Retyped's was= types), "any" accepts everything (fields behind a Rewrite).
_FieldExpectation: TypeAlias = tuple[Literal["exact", "shape", "any"], JsonValue]
_ANY: _FieldExpectation = ("any", None)


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
    kind, value = expectation
    if kind == "any":
        return None
    return _shape_of(value) if kind == "exact" else value


@dataclasses.dataclass(frozen=True, slots=True)
class _Generation:
    start: int  # steps[start:] lead from this generation to the current schema
    class_name: str
    expectations: Mapping[str, _FieldExpectation]


class _WalkError(Exception):
    pass


def _walk_generations(
    *,
    owner: str,
    steps: tuple[MigrationStep, ...],
    class_name: str,
    current_fields: Mapping[str, _FieldExpectation],
    shape_for: Callable[[Retyped], _FieldExpectation],
) -> tuple[list[_Generation], dict[int, str]]:
    """Walk the changelog newest-first, reconstructing each older generation."""
    expectations = dict(current_fields)
    current_name_of = {name: name for name in expectations}
    added_current_name: dict[int, str] = {}
    generations: list[_Generation] = []
    class_at = class_name

    def fail(index: int, message: str) -> _WalkError:
        return _WalkError(
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
                # A rewrite reshapes values within fields; it preserves the
                # field-name set, so everything else about earlier schemas is
                # unknown but the names still walk.
                expectations = dict.fromkeys(expectations, _ANY)
        generations.append(
            _Generation(
                start=index, class_name=class_at, expectations=dict(expectations)
            )
        )
    generations.reverse()
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
            owner=cls.__name__,
            steps=steps,
            class_name="",
            current_fields=dict.fromkeys(fields_by_name, _ANY),
            shape_for=lambda step: _ANY,
        )
    except _WalkError as error:
        raise TypeError(str(error)) from None

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


@dataclasses.dataclass(frozen=True, slots=True)
class _ClassResolution:
    steps: tuple[MigrationStep, ...]
    added_current_name: Mapping[int, str]
    covered: tuple[tuple[_Generation, Path], ...]  # most recent generation first
    orphaned: tuple[Path, ...]  # schema directories with no chain to current


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
    for name, (kind, value) in generation.expectations.items():
        if kind == "exact" and snapshot_fields[name] != value:
            return False
        if kind == "shape" and _shape_of(snapshot_fields[name]) != value:
            return False
    return True


def _reject_ambiguous_chains(cls: type, generations: list[_Generation]) -> None:
    described: dict[str, int] = {}
    for generation in generations:
        fields_key: dict[str, JsonValue] = {
            name: [kind, value]
            for name, (kind, value) in generation.expectations.items()
        }
        key = _stable_json_dump({"class": generation.class_name, "fields": fields_key})
        first = described.setdefault(key, generation.start)
        if first != generation.start:
            raise MigrationError(
                f"{cls.__name__}.migrations is ambiguous: the chains starting at "
                f"steps[{first}] and steps[{generation.start}] describe the same "
                "source schema; every source schema must have exactly one chain "
                "to the current schema"
            )


def _resolve_class(obj: Spec[Any]) -> _ClassResolution:
    cls = type(obj)
    steps = cls.migrations
    current_schema = cast("dict[str, JsonValue]", obj._schema_data)
    current_class = cast(str, current_schema[CLASSMARKER])
    current_fields: dict[str, _FieldExpectation] = {
        name: ("exact", schema)
        for name, schema in cast(
            "dict[str, JsonValue]", current_schema[FIELDSMARKER]
        ).items()
    }

    def shape_for(step: Retyped) -> _FieldExpectation:
        schema = schema_type(
            step.was, set(), artifact_serializers=obj.artifact_serializers
        )
        return ("shape", _shape_of(schema))

    try:
        generations, added_current_name = _walk_generations(
            owner=cls.__name__,
            steps=steps,
            class_name=current_class,
            current_fields=current_fields,
            shape_for=shape_for,
        )
    except _WalkError as error:
        raise MigrationError(str(error)) from None

    _reject_ambiguous_chains(cls, generations)

    for index, step in enumerate(steps):
        if not isinstance(step, Retyped):
            continue
        post = (
            generations[index + 1].expectations
            if index + 1 < len(steps)
            else current_fields
        )
        post_shape = _shape_of_expectation(post[step.field])
        was_shape = _shape_of_expectation(shape_for(step))
        if post_shape is not None and post_shape == was_shape:
            raise MigrationError(
                f"{cls.__name__}.migrations[{index}] ({_describe_step(step)}) is a "
                f"dead step: {step.field!r} already has that type, so the step can "
                "never migrate anything. If old stored values need reshaping - "
                "say a field's class itself changed - that is Rewrite's job."
            )

    covered: list[tuple[_Generation, Path]] = []
    orphaned: list[Path] = []
    trees = {current_class} | {generation.class_name for generation in generations}
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
            if matches:
                covered.append((matches[0], schema_directory))
            else:
                orphaned.append(schema_directory)
    # Prefer the most recent generation when searching for a source; every
    # candidate must pass the exact per-artifact field match either way.
    covered.sort(key=lambda pair: pair[0].start, reverse=True)
    return _ClassResolution(
        steps=steps,
        added_current_name=added_current_name,
        covered=tuple(covered),
        orphaned=tuple(orphaned),
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
