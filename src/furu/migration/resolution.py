from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from furu._declared_types import cached_type_hints
from furu.constants import CLASSMARKER, FIELDSMARKER, KINDMARKER
from furu.locking import read_text_or_none
from furu.migration.steps import (
    Added,
    MigrationError,
    MigrationStep,
    MovedFrom,
    Renamed,
    Retyped,
    Rewrite,
    _commutes,
    _describe_step,
    _is_breaking,
    _step_identity,
    validate_migration_declaration,
)
from furu.serializer.artifact import to_json
from furu.serializer.registry import Serializer
from furu.serializer.schema import schema_type
from furu.storage._layout import migrations_path_in, schema_snapshot_path_in
from furu.utils import (
    JsonFields,
    JsonValue,
    _stable_json_dump,
    fully_qualified_name,
)

if TYPE_CHECKING:
    from furu.core import Spec

type _FieldExpectation = tuple[Literal["exact", "shape", "any"], JsonValue]
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
    pending: tuple[int, ...]  # indices into the chain's steps, applied in order
    breaking: bool  # a pending step discards old results instead of migrating
    class_name: str
    expectations: Mapping[str, _FieldExpectation]
    pinned: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _Chain:
    """One class's migration chain, resolved against its current schema."""

    class_name: str
    names: frozenset[str]  # the current name and every name it moved from
    steps: tuple[MigrationStep, ...]
    identities: tuple[tuple[str, ...], ...]  # _step_identity of each step
    added_types: Mapping[int, object]  # declared type of each non-breaking Added
    artifact_serializers: tuple[type[Serializer], ...]
    current_schema: JsonValue
    # The old schema that the given pending steps migrate to current.
    source: Callable[[tuple[int, ...]], _Generation]

    @property
    def label(self) -> str:
        return self.class_name.rsplit(".", 1)[-1]

    def pending(self, had: set[tuple[str, ...]]) -> tuple[int, ...]:
        return tuple(
            index
            for index, identity in enumerate(self.identities)
            if identity not in had
        )

    def clash(self, had: set[tuple[str, ...]]) -> str | None:
        """Why the steps a schema lacks can't run after the ones it has, if so."""
        for index in self.pending(had):
            for later in range(index + 1, len(self.steps)):
                if self.identities[later] in had and not _commutes(
                    self.steps[index], self.steps[later]
                ):
                    return (
                        f"{self.label}.migrations[{index}] "
                        f"({_describe_step(self.steps[index])}) would have to run "
                        f"after migrations[{later}] "
                        f"({_describe_step(self.steps[later])}), which this "
                        "schema already has, and the two conflict"
                    )
        return None


@dataclass(frozen=True, slots=True)
class _ChildMove:
    chain: _Chain
    generation: _Generation


@dataclass(frozen=True, slots=True)
class _Covered:
    generation: _Generation
    child_moves: Mapping[str, _ChildMove]  # keyed by the source schema's class name
    schema_directory: Path


@dataclass(frozen=True, slots=True)
class _Orphaned:
    schema_directory: Path
    clash: str | None  # set when branches made conflicting steps in parallel


@dataclass(frozen=True, slots=True)
class _ClassResolution:
    own: _Chain
    covered: tuple[_Covered, ...]
    orphaned: tuple[_Orphaned, ...]
    # (class name, *step identity) for every step in own's and embedded
    # classes' chains; recorded beside each schema snapshot.
    declared_steps: frozenset[tuple[str, ...]]


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

    ``schema_type`` records every class it visits as a key of ``seen``, so this
    is by construction the set of classes that actually appear in the schema. cls's
    field types are walked directly so cls's own name (which need not be
    importable when this runs at class definition) is never serialized.
    """
    assert is_dataclass(cls)
    seen: dict[type, bool] = {cls: True}
    hints = cached_type_hints(cls)
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


def validate_embedded_migration_declarations(cls: type[Spec]) -> None:
    try:
        children = _embedded_migratable_classes(cls, cls.artifact_serializers)
    except Exception:  # noqa: BLE001
        return  # schema not buildable yet (forward references, ...); resolution re-checks
    for child in children:
        validate_migration_declaration(cast("type[Spec]", child))


def _build_chain(
    cls: type, artifact_serializers: tuple[type[Serializer], ...]
) -> _Chain:
    steps = tuple(cast("tuple[MigrationStep, ...]", getattr(cls, "migrations", ())))
    class_name = fully_qualified_name(cls)
    current_schema = cast(
        "dict[str, JsonValue]",
        schema_type(cls, {}, artifact_serializers=artifact_serializers),
    )
    current_fields: dict[str, _FieldExpectation] = {
        name: ("exact", schema)
        for name, schema in cast(
            "dict[str, JsonValue]", current_schema[FIELDSMARKER]
        ).items()
    }

    hints = cached_type_hints(cls)
    added_types: dict[int, object] = {}

    @functools.cache
    def source(pending: tuple[int, ...]) -> _Generation:
        expectations = dict(current_fields)
        current_name_of = {name: name for name in expectations}
        pinned: dict[str, JsonValue] = {}
        rewritten = False
        class_at = class_name
        for index in reversed(pending):
            match steps[index]:
                case Renamed(field=field, to=to):
                    expectations[field] = expectations.pop(to)
                    current_name_of[field] = current_name_of.pop(to)
                case Added(field=field) as step:
                    name = current_name_of.pop(field)
                    del expectations[field]
                    if not step.breaking:
                        added_types[index] = hints[name]
                        if not rewritten and step.default_factory is None:
                            pinned[name] = to_json(
                                step.default,
                                declared_type=hints[name],
                                artifact_serializers=artifact_serializers,
                            )
                case Retyped(field=field) as step:
                    was_shape = _shape_of(
                        schema_type(
                            step.was,
                            {},
                            artifact_serializers=artifact_serializers,
                        )
                    )
                    kind, value = expectations[field]
                    if kind != "any" and was_shape == (
                        _shape_of(value) if kind == "exact" else value
                    ):
                        raise MigrationError(
                            f"{cls.__name__}.migrations[{index}] "
                            f"({_describe_step(step)}) is a dead step: {field!r} "
                            "already has that type, so the step can never migrate "
                            "anything. If old stored values need reshaping - say a "
                            "field's class itself changed - that is Rewrite's job."
                        )
                    expectations[field] = ("shape", was_shape)
                case MovedFrom(fully_qualified_name=name):
                    class_at = name
                case Rewrite():
                    expectations = dict.fromkeys(expectations, _ANY)
                    rewritten = True
        return _Generation(
            pending=pending,
            breaking=any(_is_breaking(steps[index]) for index in pending),
            class_name=class_at,
            expectations=expectations,
            pinned=pinned,
        )

    # Walk the whole chain once: it checks for dead steps and fills added_types.
    source(tuple(range(len(steps))))
    return _Chain(
        class_name=class_name,
        names=frozenset(
            (
                class_name,
                *(s.fully_qualified_name for s in steps if isinstance(s, MovedFrom)),
            )
        ),
        steps=steps,
        identities=tuple(_step_identity(step) for step in steps),
        added_types=added_types,
        artifact_serializers=artifact_serializers,
        current_schema=current_schema,
        source=source,
    )


def _normalize_snapshot(
    snapshot: JsonValue,
    chains: tuple[_Chain, ...],
    generations: Mapping[str, _Generation],
) -> tuple[JsonValue, Mapping[str, _ChildMove]]:
    """Rewrite embedded old-generation sub-schemas to their current spelling.

    ``generations`` holds, per chain, the old schema the stored steps point to.
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
            _ChildMove(chain, generation)
            for chain in chains
            if (generation := generations.get(chain.class_name)) is not None
            and generation.pending
            and _snapshot_matches(out, generation)
        ]
        if not matches:
            return out
        if len(matches) > 1:
            raise MigrationError(
                f"embedded migrations are ambiguous: the recorded sub-schema for "
                f"{out.get(CLASSMARKER)!r} matches both {matches[0].chain.label} "
                f"and {matches[1].chain.label}; every source schema must have "
                "exactly one chain to the current schema"
            )
        (move,) = matches
        if moves.setdefault(move.generation.class_name, move) != move:
            raise MigrationError(
                f"embedded migrations are ambiguous: {move.generation.class_name!r} "
                "is claimed by two different classes' chains, so its stored "
                "values cannot be replayed uniformly"
            )
        return move.chain.current_schema

    normalized = normalize(snapshot)
    return normalized, dict(sorted(moves.items()))


def _recorded_steps(
    schema_directory: Path, chains: tuple[_Chain, ...]
) -> dict[str, set[tuple[str, ...]]] | None:
    """The steps each chain's class had when this schema was written, or None
    if the writing code took a step these chains never declare - the schema
    then belongs to a sibling or newer branch sharing the store."""
    had: dict[str, set[tuple[str, ...]]] = {chain.class_name: set() for chain in chains}
    recorded = migrations_path_in(schema_directory)
    for name, *identity in json.loads(read_text_or_none(recorded) or "[]"):
        chain = next((chain for chain in chains if name in chain.names), None)
        if chain is None or tuple(identity) not in chain.identities:
            return None
        had[chain.class_name].add(tuple(identity))
    return had


def _resolve_class(obj: Spec) -> _ClassResolution:
    cls = type(obj)
    own = _build_chain(cls, obj.artifact_serializers)
    children = _embedded_migratable_classes(cls, obj.artifact_serializers)
    for child in children:
        validate_migration_declaration(cast("type[Spec]", child))
    chains = tuple(_build_chain(child, obj.artifact_serializers) for child in children)
    declared_steps = frozenset(
        (chain.class_name, *identity)
        for chain in (own, *chains)
        for identity in chain.identities
    )

    covered: list[_Covered] = []
    orphaned: list[_Orphaned] = []
    for tree_name in sorted(own.names):
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
            snapshot_path = schema_snapshot_path_in(schema_directory)
            if not snapshot_path.exists():
                continue
            had = _recorded_steps(schema_directory, (own, *chains))
            if had is None:
                continue
            pending = own.pending(had[own.class_name])
            if any(_is_breaking(own.steps[index]) for index in pending):
                continue  # superseded: recomputed, neither reused nor stale
            if clash := own.clash(had[own.class_name]):
                orphaned.append(_Orphaned(schema_directory, clash))
                continue
            clashes: list[str] = []
            child_generations: dict[str, _Generation] = {}
            for chain in chains:
                if child_clash := chain.clash(had[chain.class_name]):
                    clashes.append(child_clash)
                else:
                    child_generations[chain.class_name] = chain.source(
                        chain.pending(had[chain.class_name])
                    )
            snapshot = cast(
                JsonValue, json.loads(snapshot_path.read_text(encoding="utf-8"))
            )
            child_moves: Mapping[str, _ChildMove] = {}
            if chains:
                snapshot, child_moves = _normalize_snapshot(
                    snapshot, chains, child_generations
                )
            generation = own.source(pending)
            if not _snapshot_matches(snapshot, generation):
                orphaned.append(_Orphaned(schema_directory, "; ".join(clashes) or None))
            elif not any(move.generation.breaking for move in child_moves.values()):
                covered.append(
                    _Covered(
                        generation=generation,
                        child_moves=child_moves,
                        schema_directory=schema_directory,
                    )
                )
    covered.sort(key=lambda entry: len(entry.generation.pending))
    return _ClassResolution(
        own=own,
        covered=tuple(covered),
        orphaned=tuple(orphaned),
        declared_steps=declared_steps,
    )


_RESOLUTION_CACHE: dict[tuple[type, Path], _ClassResolution | MigrationError] = {}


def _class_resolution(obj: Spec) -> _ClassResolution:
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


def _apply_steps(
    chain: _Chain, generation: _Generation, source_fields: JsonFields
) -> JsonFields:
    fields = dict(source_fields)
    for index in generation.pending:
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
            case Added(field=field, default=default, default_factory=factory):
                fields[field] = to_json(
                    factory(_SourceFields(fields, description)) if factory else default,
                    declared_type=chain.added_types[index],
                    artifact_serializers=chain.artifact_serializers,
                )
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
                    move.chain, move.generation, cast(JsonFields, out[FIELDSMARKER])
                ),
            )
    return out
