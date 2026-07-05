from __future__ import annotations

import json
import typing
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from furu.constants import CLASSMARKER, FIELDSMARKER, KINDMARKER
from furu.migration.steps import (
    Added,
    MigrationError,
    MigrationStep,
    MovedFrom,
    Renamed,
    Retyped,
    Rewrite,
    _describe_step,
    _is_breaking,
    validate_migration_declaration,
)
from furu.serializer.artifact import to_json
from furu.serializer.registry import Serializer
from furu.serializer.schema import schema_type
from furu.storage._layout import schema_snapshot_path_in_schema_directory
from furu.utils import (
    JsonFields,
    JsonValue,
    _stable_json_dump,
    fully_qualified_name,
)

if TYPE_CHECKING:
    from furu.core import Spec

_FieldExpectation: TypeAlias = tuple[Literal["exact", "shape", "any"], JsonValue]
_ANY: _FieldExpectation = ("any", None)


def _shape_of(schema: JsonValue) -> JsonValue:
    if isinstance(schema, dict):
        if CLASSMARKER in schema:
            return schema[CLASSMARKER]
        return {key: _shape_of(value) for key, value in schema.items()}
    if isinstance(schema, list):
        return sorted((_shape_of(item) for item in schema), key=_stable_json_dump)
    return schema


@dataclass(frozen=True, slots=True)
class _Generation:
    start: int
    class_name: str
    expectations: Mapping[str, _FieldExpectation]


@dataclass(frozen=True, slots=True)
class _Chain:
    """One class's migration chain, resolved against its current schema.

    ``generations[i]`` describes the source schema that ``steps[i:]`` migrates
    to current; ``generations[-1]`` is the current schema itself.
    """

    class_name: str
    steps: tuple[MigrationStep, ...]
    generations: tuple[_Generation, ...]
    last_breaking: int
    added_defaults: Mapping[int, JsonValue]
    current_schema: JsonValue

    @property
    def label(self) -> str:
        return self.class_name.rsplit(".", 1)[-1]


@dataclass(frozen=True, slots=True)
class _ChildMove:
    chain: _Chain
    start: int


@dataclass(frozen=True, slots=True)
class _Covered:
    generation: _Generation
    child_moves: Mapping[str, _ChildMove]  # keyed by the source schema's class name
    schema_directory: Path


@dataclass(frozen=True, slots=True)
class _ClassResolution:
    own: _Chain
    covered: tuple[_Covered, ...]
    orphaned: tuple[Path, ...]


def _snapshot_matches(snapshot: JsonValue, generation: _Generation) -> bool:
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get(CLASSMARKER) != generation.class_name:
        return False
    fields = snapshot.get(FIELDSMARKER)
    if not isinstance(fields, dict) or set(fields) != set(generation.expectations):
        return False
    return all(
        (kind != "exact" or fields[name] == value)
        and (kind != "shape" or _shape_of(fields[name]) == value)
        for name, (kind, value) in generation.expectations.items()
    )


def _embedded_migratable_classes(
    cls: type, artifact_serializers: tuple[type[Serializer], ...]
) -> tuple[type, ...]:
    """Classes embedded in cls's schema that declare a migration chain.

    ``schema_type`` records every class it visits in ``seen``, so this is by
    construction the set of classes that actually appear in the schema. cls's
    field types are walked directly so cls's own name (which need not be
    importable when this runs at class definition) is never serialized.
    """
    seen: set[type] = {cls}
    hints = typing.get_type_hints(cls, include_extras=True)
    for field in dataclass_fields(cls):
        schema_type(hints[field.name], seen, artifact_serializers=artifact_serializers)
    return tuple(
        sorted(
            (
                tp
                for tp in seen
                if tp is not cls
                and is_dataclass(tp)
                and getattr(tp, "migrations", None)
            ),
            key=fully_qualified_name,
        )
    )


def validate_embedded_migration_declarations(cls: type[Spec[Any]]) -> None:
    try:
        children = _embedded_migratable_classes(cls, cls.artifact_serializers)
    except Exception:
        return  # schema not buildable yet (forward references, ...); resolution re-checks
    for child in children:
        validate_migration_declaration(cast("type[Spec[Any]]", child))


def _build_chain(
    cls: type, artifact_serializers: tuple[type[Serializer], ...]
) -> _Chain:
    steps = tuple(cast("tuple[MigrationStep, ...]", getattr(cls, "migrations", ())))
    class_name = fully_qualified_name(cls)
    current_schema = cast(
        "dict[str, JsonValue]",
        schema_type(cls, set(), artifact_serializers=artifact_serializers),
    )
    current_fields: dict[str, _FieldExpectation] = {
        name: ("exact", schema)
        for name, schema in cast(
            "dict[str, JsonValue]", current_schema[FIELDSMARKER]
        ).items()
    }

    expectations = dict(current_fields)
    current_name_of = {name: name for name in expectations}
    added_current_name: dict[int, str] = {}
    generations: list[_Generation] = []
    class_at = class_name
    for index in reversed(range(len(steps))):
        match steps[index]:
            case Renamed(field=field, to=to):
                expectations[field] = expectations.pop(to)
                current_name_of[field] = current_name_of.pop(to)
            case Added(field=field) as step:
                name = current_name_of.pop(field)
                if not step.breaking:
                    added_current_name[index] = name
                del expectations[field]
            case Retyped(field=field) as step:
                expectations[field] = (
                    "shape",
                    _shape_of(
                        schema_type(
                            step.was,
                            set(),
                            artifact_serializers=artifact_serializers,
                        )
                    ),
                )
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
    generations.append(
        _Generation(
            start=len(steps), class_name=class_name, expectations=current_fields
        )
    )

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
        kind, value = generations[index + 1].expectations[step.field]
        if kind == "any":
            continue
        post_shape = _shape_of(value) if kind == "exact" else value
        _, was_shape = generations[index].expectations[step.field]
        if post_shape == was_shape:
            raise MigrationError(
                f"{cls.__name__}.migrations[{index}] ({_describe_step(step)}) is a "
                f"dead step: {step.field!r} already has that type, so the step can "
                "never migrate anything. If old stored values need reshaping - "
                "say a field's class itself changed - that is Rewrite's job."
            )

    added_defaults: dict[int, JsonValue] = {}
    if added_current_name:
        hints = typing.get_type_hints(cls, include_extras=True)
        for index, name in added_current_name.items():
            added_defaults[index] = to_json(
                cast(Added, steps[index]).default,
                declared_type=hints[name],
                artifact_serializers=artifact_serializers,
            )

    return _Chain(
        class_name=class_name,
        steps=steps,
        generations=tuple(generations),
        last_breaking=max(
            (index for index, step in enumerate(steps) if _is_breaking(step)),
            default=-1,
        ),
        added_defaults=added_defaults,
        current_schema=current_schema,
    )


def _normalize_snapshot(
    snapshot: JsonValue, chains: tuple[_Chain, ...]
) -> tuple[JsonValue, Mapping[str, _ChildMove]]:
    """Rewrite embedded old-generation sub-schemas to their current spelling.

    Returns the normalized snapshot plus one move per recognized source class
    name; the caller replays those moves on the stored values.
    """
    moves: dict[str, _ChildMove] = {}

    def normalize(node: JsonValue) -> JsonValue:
        if isinstance(node, list):
            # Union schemas are stored sorted; normalization can reorder them.
            return sorted((normalize(item) for item in node), key=_stable_json_dump)
        if not isinstance(node, dict) or node.get(KINDMARKER) == "custom":
            return node
        out = {key: normalize(value) for key, value in node.items()}
        if FIELDSMARKER not in out or any(
            out == chain.current_schema for chain in chains
        ):
            # An already-current sub-schema must not match a chain generation:
            # a Rewrite generation expects any field schema, current included.
            return out
        matches = [
            (chain, generation)
            for chain in chains
            for generation in chain.generations[:-1]
            if _snapshot_matches(out, generation)
        ]
        if not matches:
            return out
        if len(matches) > 1:
            (first_chain, first), (second_chain, second) = matches[:2]
            raise MigrationError(
                f"embedded migrations are ambiguous: the recorded sub-schema for "
                f"{out.get(CLASSMARKER)!r} matches "
                f"{first_chain.label}.migrations[{first.start}:] and "
                f"{second_chain.label}.migrations[{second.start}:]; every source "
                "schema must have exactly one chain to the current schema"
            )
        (chain, generation) = matches[0]
        move = _ChildMove(chain=chain, start=generation.start)
        if moves.setdefault(generation.class_name, move) != move:
            raise MigrationError(
                f"embedded migrations are ambiguous: {generation.class_name!r} "
                "appears in the recorded schema at two different chain positions, "
                "so its stored values cannot be replayed uniformly"
            )
        return chain.current_schema

    normalized = normalize(snapshot)
    return normalized, dict(sorted(moves.items()))


def _resolve_class(obj: Spec[Any]) -> _ClassResolution:
    cls = type(obj)
    own = _build_chain(cls, obj.artifact_serializers)
    children = _embedded_migratable_classes(cls, obj.artifact_serializers)
    for child in children:
        validate_migration_declaration(cast("type[Spec[Any]]", child))
    chains = tuple(_build_chain(child, obj.artifact_serializers) for child in children)

    covered: list[_Covered] = []
    orphaned: list[Path] = []
    trees = {generation.class_name for generation in own.generations}
    for tree_name in sorted(trees):
        tree_directory = obj._metadata.storage / Path(*tree_name.split("."))
        if not tree_directory.exists():
            continue
        for schema_directory in sorted(
            path for path in tree_directory.iterdir() if path.is_dir()
        ):
            if (
                tree_name == own.class_name
                and schema_directory.name == obj._artifact_schema_hash
            ):
                continue
            snapshot_path = schema_snapshot_path_in_schema_directory(schema_directory)
            if not snapshot_path.exists():
                continue
            snapshot = cast(
                JsonValue, json.loads(snapshot_path.read_text(encoding="utf-8"))
            )
            child_moves: Mapping[str, _ChildMove] = {}
            if chains:
                snapshot, child_moves = _normalize_snapshot(snapshot, chains)
            matches = [
                generation
                for generation in own.generations
                if _snapshot_matches(snapshot, generation)
            ]
            if len(matches) > 1:
                raise MigrationError(
                    f"{cls.__name__}.migrations is ambiguous: the recorded schema "
                    f"at {schema_directory} matches the chains starting at "
                    f"steps[{matches[0].start}] and steps[{matches[1].start}]; "
                    "every source schema must have exactly one chain to the "
                    "current schema"
                )
            if not matches:
                orphaned.append(schema_directory)
            elif matches[0].start > own.last_breaking and all(
                move.start > move.chain.last_breaking for move in child_moves.values()
            ):
                covered.append(
                    _Covered(
                        generation=matches[0],
                        child_moves=child_moves,
                        schema_directory=schema_directory,
                    )
                )
    covered.sort(key=lambda entry: entry.generation.start, reverse=True)
    return _ClassResolution(
        own=own,
        covered=tuple(covered),
        orphaned=tuple(orphaned),
    )


_RESOLUTION_CACHE: dict[tuple[type, Path], _ClassResolution | MigrationError] = {}


def _class_resolution(obj: Spec[Any]) -> _ClassResolution:
    key = (type(obj), obj._metadata.storage)
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


def _apply_steps(chain: _Chain, start: int, source_fields: JsonFields) -> JsonFields:
    fields = dict(source_fields)
    for index in range(start, len(chain.steps)):
        step = chain.steps[index]
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
                fields[field] = chain.added_defaults[index]
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


def _apply_child_moves(value: JsonValue, moves: Mapping[str, _ChildMove]) -> JsonValue:
    """Replay matched child chains over every embedded instance, innermost first."""
    if isinstance(value, list):
        return [_apply_child_moves(item, moves) for item in value]
    if not isinstance(value, dict) or value.get(KINDMARKER) == "custom":
        return value
    out = {key: _apply_child_moves(item, moves) for key, item in value.items()}
    if out.get(KINDMARKER) == "instance":
        name = out.get(CLASSMARKER)
        if isinstance(name, str) and (move := moves.get(name)) is not None:
            out[CLASSMARKER] = move.chain.class_name
            out[FIELDSMARKER] = cast(
                JsonValue,
                _apply_steps(
                    move.chain, move.start, cast(JsonFields, out[FIELDSMARKER])
                ),
            )
    return out
