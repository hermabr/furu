from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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
from furu.utils import JsonFields, JsonValue, _stable_json_dump

if TYPE_CHECKING:
    from furu.core import Spec

# A field's expected schema at some point in the chain: matched exactly for
# untouched fields, at shape level (class identity, not internal layout) behind
# Retyped, and not at all behind Rewrite.
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


def _was_shape(obj: Spec[Any], step: Retyped) -> JsonValue:
    schema = schema_type(step.was, set(), artifact_serializers=obj.artifact_serializers)
    return _shape_of(schema)


@dataclass(frozen=True, slots=True)
class _Generation:
    start: int  # steps[start:] lead from this generation to the current schema
    class_name: str
    expectations: Mapping[str, _FieldExpectation]


@dataclass(frozen=True, slots=True)
class _ClassResolution:
    steps: tuple[MigrationStep, ...]
    added_current_name: dict[int, str]
    covered: tuple[tuple[_Generation, Path], ...]  # most recent generation first
    orphaned: tuple[Path, ...]  # schema directories with no chain to current


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


def _added_default_fields(
    obj: Spec[Any], resolution: _ClassResolution
) -> dict[int, JsonValue]:
    if not resolution.added_current_name:
        return {}
    field_by_name = {field.name: field for field in dataclasses.fields(type(obj))}
    replacements: dict[str, Any] = {}
    for current_name in set(resolution.added_current_name.values()):
        field = field_by_name[current_name]
        if field.default is not dataclasses.MISSING:
            replacements[current_name] = field.default
        else:
            factory = field.default_factory
            assert callable(factory)  # phase-one validation guarantees a default
            replacements[current_name] = factory()
    defaults_obj = dataclasses.replace(obj, **replacements)
    fields_json = cast(JsonFields, defaults_obj._artifact_data[FIELDSMARKER])
    return {
        index: fields_json[current_name]
        for index, current_name in resolution.added_current_name.items()
    }


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

    # Walk backward from the current schema; phase-one validation already
    # guaranteed the chain is well-formed. generations[i] is the source schema
    # of the chain steps[i:].
    expectations = dict(current_fields)
    current_name_of = {name: name for name in expectations}
    added_current_name: dict[int, str] = {}
    generations: list[_Generation] = []
    class_at = current_class
    for index in reversed(range(len(steps))):
        match steps[index]:
            case Renamed(field=field, to=to):
                expectations[field] = expectations.pop(to)
                current_name_of[field] = current_name_of.pop(to)
            case Added(field=field):
                added_current_name[index] = current_name_of.pop(field)
                del expectations[field]
            case Retyped(field=field) as step:
                expectations[field] = ("shape", _was_shape(obj, step))
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

    for position, generation in enumerate(generations):
        for other in generations[position + 1 :]:
            if (
                generation.class_name == other.class_name
                and generation.expectations == other.expectations
            ):
                raise MigrationError(
                    f"{cls.__name__}.migrations is ambiguous: the chains starting "
                    f"at steps[{generation.start}] and steps[{other.start}] "
                    "describe the same source schema; every source schema must "
                    "have exactly one chain to the current schema"
                )

    for index, step in enumerate(steps):
        if not isinstance(step, Retyped):
            continue
        post = (
            generations[index + 1].expectations
            if index + 1 < len(steps)
            else current_fields
        )
        kind, value = post[step.field]
        if kind == "any":
            continue
        post_shape = _shape_of(value) if kind == "exact" else value
        if post_shape == _was_shape(obj, step):
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
    # Prefer the most recent generation; every candidate still matches fields exactly.
    covered.sort(key=lambda pair: pair[0].start, reverse=True)
    return _ClassResolution(
        steps=steps,
        added_current_name=added_current_name,
        covered=tuple(covered),
        orphaned=tuple(orphaned),
    )


_RESOLUTION_CACHE: dict[tuple[type, Path], _ClassResolution | MigrationError] = {}


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


def _apply_steps(
    resolution: _ClassResolution,
    start: int,
    source_fields: JsonFields,
    added_defaults: dict[int, JsonValue],
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
                fields[field] = added_defaults[index]
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
