from __future__ import annotations

import dataclasses
import json
import types
import typing
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict

from furu.constants import CLASSMARKER, FIELDSMARKER
from furu.metadata import CompletedMetadata
from furu.serializer.schema import schema_type
from furu.storage._layout import (
    metadata_path_in,
    result_dir_in,
    result_link_path_in,
    result_manifest_path_in,
    schema_snapshot_path_in_schema_directory,
)
from furu.utils import (
    JsonFields,
    JsonValue,
    _stable_json_dump,
    fully_qualified_name,
    nfs_safe_unique_name,
)

if TYPE_CHECKING:
    from furu.core import Spec


@dataclass(frozen=True, slots=True)
class Renamed:
    field: str
    _: dataclasses.KW_ONLY
    to: str


@dataclass(frozen=True, slots=True)
class Added:
    field: str


@dataclass(frozen=True, slots=True)
class MovedFrom:
    fully_qualified_name: str


@dataclass(frozen=True, slots=True)
class Retyped:
    field: str
    _: dataclasses.KW_ONLY
    was: Any


@dataclass(frozen=True, slots=True)
class Rewrite:
    transform: Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]


MigrationStep: TypeAlias = Renamed | Added | MovedFrom | Retyped | Rewrite


class Stale(Exception):
    pass


class MigrationError(Exception):
    pass


def _type_label(tp: object) -> str:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        return " | ".join(_type_label(a) for a in typing.get_args(tp))
    if isinstance(tp, type):
        return fully_qualified_name(tp)
    return repr(tp)


def _describe_step(step: MigrationStep) -> str:
    match step:
        case Renamed(field=field, to=to):
            return f"Renamed({field!r}, to={to!r})"
        case Added(field=field):
            return f"Added({field!r})"
        case MovedFrom(fully_qualified_name=name):
            return f"MovedFrom({name!r})"
        case Retyped(field=field, was=was):
            return f"Retyped({field!r}, was={_type_label(was)})"
        case Rewrite(transform=transform):
            return f"Rewrite({getattr(transform, '__qualname__', repr(transform))})"


# --- source-schema prediction -------------------------------------------------
#
# The chain is a changelog: steps in declaration order lead from the oldest
# source schema to the current schema, and each suffix steps[start:] therefore
# describes one older generation. Generations are predicted by inverting the
# steps backward from the current schema; recorded schema.json snapshots are
# then bound to generations by content, never by recomputing hashes.


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


# --- phase two: resolution against recorded snapshots --------------------------


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
                "never migrate anything. If old stored values need reshaping — "
                "say a field's class itself changed — that is Rewrite's job."
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


# --- applying a chain to stored fields ------------------------------------------


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


# --- result links ---------------------------------------------------------------


class _ResultLinkCurrent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    fully_qualified_name: str
    schema_hash: str
    artifact_hash: str
    fields: JsonFields


class _ResultLinkSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    fully_qualified_name: str
    schema_hash: str
    artifact_hash: str
    base_dir: Path


class _ResultLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    current: _ResultLinkCurrent
    source: _ResultLinkSource
    migration_path: tuple[str, ...]


def _read_source(artifact_dir: Path) -> _ResultLink | None:
    result_manifest = result_manifest_path_in(artifact_dir)
    metadata_path = metadata_path_in(artifact_dir)
    if result_manifest.exists() and metadata_path.exists():
        metadata = CompletedMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        artifact = metadata.artifact
        return _ResultLink(
            current=_ResultLinkCurrent(
                fully_qualified_name=artifact.fully_qualified_name,
                schema_hash=artifact.schema_hash,
                artifact_hash=artifact.artifact_hash,
                fields=cast(JsonFields, artifact.artifact_data[FIELDSMARKER]),
            ),
            source=_ResultLinkSource(
                fully_qualified_name=artifact.fully_qualified_name,
                schema_hash=artifact.schema_hash,
                artifact_hash=artifact.artifact_hash,
                base_dir=artifact_dir,
            ),
            migration_path=(),
        )
    link_path = result_link_path_in(artifact_dir)
    if link_path.exists():
        return _ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
    return None


def _find_source(obj: Spec[Any], resolution: _ClassResolution) -> _ResultLink | None:
    if not resolution.generations:
        return None
    covered = [
        directory
        for directory in resolution.directories
        if directory.generation is not None
    ]
    if not covered:
        return None
    # Prefer the most recent generation; every candidate must pass the exact
    # per-artifact field match either way, so any hit is a correct result.
    covered.sort(
        key=lambda directory: cast(_Generation, directory.generation).start,
        reverse=True,
    )
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    added_defaults = _added_default_fields(obj, resolution)
    for directory in covered:
        generation = cast(_Generation, directory.generation)
        if not directory.schema_directory.exists():
            continue
        for artifact_dir in sorted(directory.schema_directory.iterdir()):
            if not artifact_dir.is_dir():
                continue
            source_link = _read_source(artifact_dir)
            if source_link is None:
                continue
            fields = _apply_steps(
                resolution, generation.start, source_link.current.fields, added_defaults
            )
            if fields != target_fields:
                continue
            return _ResultLink(
                current=_ResultLinkCurrent(
                    fully_qualified_name=obj._fully_qualified_name,
                    schema_hash=obj._artifact_schema_hash,
                    artifact_hash=obj._artifact_hash,
                    fields=target_fields,
                ),
                source=source_link.source,
                migration_path=source_link.migration_path
                + tuple(
                    _describe_step(step)
                    for step in resolution.steps[generation.start :]
                ),
            )
    return None


def _write_result_link(obj: Spec[Any], link: _ResultLink) -> None:
    from furu.execution.load_or_create import _record_schema_snapshot

    obj._base_dir.mkdir(parents=True, exist_ok=True)
    link_path = result_link_path_in(obj._base_dir)
    tmp_path = nfs_safe_unique_name(link_path, name="tmp")
    tmp_path.write_text(link.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.rename(link_path)
    # A linked result is a result: the snapshot marks this schema directory as
    # holding results, so later chains can resolve through it.
    _record_schema_snapshot(obj)


# --- the public seams used by core and the execution paths ----------------------


def result_dir_for_loading[T](obj: Spec[T]) -> Path | None:
    if result_manifest_path_in(obj._base_dir).exists():
        return result_dir_in(obj._base_dir)
    link_path = result_link_path_in(obj._base_dir)
    if link_path.exists():
        link = _ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
        if not result_manifest_path_in(link.source.base_dir).exists():
            raise RuntimeError(f"{link_path} points to a missing result")
        return result_dir_in(link.source.base_dir)
    if not type(obj).migrations:
        return None
    link = _find_source(obj, _class_resolution(obj))
    if link is None:
        return None
    _write_result_link(obj, link)
    return result_dir_in(link.source.base_dir)


def _orphaned_directories(
    resolution: _ClassResolution,
) -> list[_GenerationDirectory]:
    # Re-stat rather than trusting the memoized scan: discarding old results is
    # directory deletion, and the block must lift the moment the directory is gone.
    return [
        directory
        for directory in resolution.directories
        if directory.generation is None and directory.snapshot_path.exists()
    ]


def sideways_status(obj: Spec[Any]) -> Literal["done", "stale", "missing"]:
    resolution = _class_resolution(obj)
    if _find_source(obj, resolution) is not None:
        return "done"
    if _orphaned_directories(resolution):
        return "stale"
    return "missing"


def raise_if_stale(obj: Spec[Any]) -> None:
    resolution = _class_resolution(obj)
    orphaned = _orphaned_directories(resolution)
    if not orphaned:
        return
    current_schema = cast("dict[str, JsonValue]", obj._schema_data)
    current_fields = cast("dict[str, JsonValue]", current_schema[FIELDSMARKER])
    lines = [
        f"{type(obj).__name__} is stale: the store holds results under "
        f"{len(orphaned)} other schema(s) with no migration chain to the "
        "current schema."
    ]
    for directory in orphaned:
        snapshot = cast(
            "dict[str, JsonValue]",
            json.loads(directory.snapshot_path.read_text(encoding="utf-8")),
        )
        lines.append(f"\norphaned: {directory.schema_directory}")
        lines.extend(
            _schema_field_diff(
                cast("dict[str, JsonValue]", snapshot.get(FIELDSMARKER, {})),
                current_fields,
            )
        )
    lines.append(
        f"\nEither declare a migration chain on {type(obj).__name__}.migrations "
        "(Renamed/Added/MovedFrom/Retyped/Rewrite), or discard the orphaned "
        "results by deleting the directory above."
    )
    raise Stale("\n".join(lines))


def _schema_field_diff(
    old_fields: dict[str, JsonValue], new_fields: dict[str, JsonValue]
) -> list[str]:
    lines: list[str] = []
    for name in sorted(old_fields.keys() | new_fields.keys()):
        if name not in old_fields:
            lines.append(f"  + {name}: {_stable_json_dump(new_fields[name])}")
        elif name not in new_fields:
            lines.append(f"  - {name}: {_stable_json_dump(old_fields[name])}")
        elif old_fields[name] != new_fields[name]:
            lines.append(
                f"  ~ {name}: {_stable_json_dump(old_fields[name])} "
                f"-> {_stable_json_dump(new_fields[name])}"
            )
    return lines
