from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from furu.constants import FIELDSMARKER
from furu.metadata import CompletedMetadata
from furu.migration.resolution import (
    _apply_child_moves,
    _apply_steps,
    _class_resolution,
    _ClassResolution,
    _SourceFields,
)
from furu.migration.steps import (
    MigrationError,
    MigrationStep,
    ResultAdded,
    ResultChange,
    ResultRemoved,
    ResultRenamed,
    ResultRewrite,
    _describe_result_change,
    _describe_step,
)
from furu.result.bundle import WRAPPER_KEY, _DumpState, _dump_value
from furu.storage._layout import (
    metadata_path_in,
    result_dir_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.utils import (
    JsonFields,
    JsonValue,
    atomic_write_text,
    resolve_fully_qualified_name,
)

if TYPE_CHECKING:
    from furu.core import Spec


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


class _ResultHop(BaseModel):
    """One class's migrations[start:stop] slice whose result_changes apply on load.

    ``stop`` pins the slice so steps appended to the chain later (which get
    their own link under the new schema hash) are never replayed twice.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    fully_qualified_name: str
    start: int
    stop: int


class _ResultLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    current: _ResultLinkCurrent
    source: _ResultLinkSource
    migration_path: tuple[str, ...]
    result_migrations: tuple[_ResultHop, ...]


def _read_source(artifact_dir: Path) -> _ResultLink | None:
    result_manifest = result_manifest_path_in(artifact_dir)
    metadata_path = metadata_path_in(artifact_dir)
    if result_manifest.exists() and metadata_path.exists():
        metadata = CompletedMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        return _ResultLink(
            current=_ResultLinkCurrent(
                fully_qualified_name=metadata.artifact.fully_qualified_name,
                schema_hash=metadata.artifact.schema_hash,
                artifact_hash=metadata.artifact.artifact_hash,
                fields=cast(JsonFields, metadata.artifact.artifact_data[FIELDSMARKER]),
            ),
            source=_ResultLinkSource(
                fully_qualified_name=metadata.artifact.fully_qualified_name,
                schema_hash=metadata.artifact.schema_hash,
                artifact_hash=metadata.artifact.artifact_hash,
                base_dir=artifact_dir,
            ),
            migration_path=(),
            result_migrations=(),
        )
    link_path = result_link_path_in(artifact_dir)
    if link_path.exists():
        return _ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
    return None


def _find_source(obj: Spec[Any], resolution: _ClassResolution) -> _ResultLink | None:
    if not resolution.covered:
        return None
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    for covered in resolution.covered:
        if not covered.schema_directory.exists():
            continue
        for artifact_dir in sorted(covered.schema_directory.iterdir()):
            if not artifact_dir.is_dir():
                continue
            source_link = _read_source(artifact_dir)
            if source_link is None:
                continue
            fields = source_link.current.fields
            if covered.child_moves:
                fields = {
                    name: _apply_child_moves(value, covered.child_moves)
                    for name, value in fields.items()
                }
            fields = _apply_steps(resolution.own, covered.generation.start, fields)
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
                    f"{move.chain.label}: {_describe_step(step)}"
                    for move in covered.child_moves.values()
                    for step in move.chain.steps[move.start :]
                )
                + tuple(
                    _describe_step(step)
                    for step in resolution.own.steps[covered.generation.start :]
                ),
                result_migrations=source_link.result_migrations
                + (
                    (
                        _ResultHop(
                            fully_qualified_name=resolution.own.class_name,
                            start=covered.generation.start,
                            stop=len(resolution.own.steps),
                        ),
                    )
                    if any(
                        step.result_changes
                        for step in resolution.own.steps[covered.generation.start :]
                    )
                    else ()
                ),
            )
    return None


def _write_result_link(obj: Spec[Any], link: _ResultLink) -> None:
    from furu.execution.load_or_create import _record_schema_snapshot

    obj._base_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        result_link_path_in(obj._base_dir), link.model_dump_json(indent=2)
    )
    _record_schema_snapshot(obj)


@dataclass(frozen=True, slots=True)
class _LoadableResult:
    result_dir: Path
    manifest_transform: Callable[[JsonValue], JsonValue] | None


def _encoded_result_value(value: object) -> JsonValue:
    unused = Path("unused")
    return _dump_value(
        value,
        declared_type=Any,
        value_path=(),
        bundle_dir=unused,
        result_codecs=(),
        dump_state=_DumpState(data_dir=unused),
    )


def _apply_result_changes(
    manifest: JsonValue,
    changes: tuple[ResultChange, ...],
    *,
    spec_fields: JsonFields,
) -> JsonValue:
    body: dict[str, JsonValue] | None = None
    raw_fields: JsonValue = manifest
    if isinstance(manifest, dict) and WRAPPER_KEY in manifest:
        body = dict(cast("dict[str, JsonValue]", manifest[WRAPPER_KEY]))
        raw_fields = body.get(FIELDSMARKER)
    if not isinstance(raw_fields, dict):
        raise MigrationError(
            "result_changes need a stored result with named fields "
            "(a dataclass, pydantic model, or dict at the top level)"
        )
    fields = dict(raw_fields)
    for change in changes:
        description = _describe_result_change(change)
        match change:
            case ResultAdded(field=field, value=value):
                fields[field] = _encoded_result_value(value)
            case ResultRenamed(field=field) | ResultRemoved(field=field) if (
                field not in fields
            ):
                raise MigrationError(
                    f"{description}: stored result has no field {field!r}; "
                    f"stored fields: {sorted(fields)}"
                )
            case ResultRenamed(field=field, to=to):
                fields[to] = fields.pop(field)
            case ResultRemoved(field=field):
                del fields[field]
            case ResultRewrite(transform=transform):
                rewritten = dict(
                    transform(
                        _SourceFields(fields, description),
                        spec=_SourceFields(dict(spec_fields), description),
                    )
                )
                if set(rewritten) != set(fields):
                    raise MigrationError(
                        f"{description} must preserve field names: it returned "
                        f"{sorted(rewritten)} for source fields {sorted(fields)}; "
                        "use ResultRenamed/ResultAdded/ResultRemoved for shape "
                        "changes"
                    )
                fields = rewritten
    if body is None:
        return cast(JsonValue, fields)
    body[FIELDSMARKER] = cast(JsonValue, fields)
    return {WRAPPER_KEY: cast(JsonValue, body)}


def _manifest_transform_for(
    link: _ResultLink,
) -> Callable[[JsonValue], JsonValue] | None:
    changes = tuple(
        change
        for hop in link.result_migrations
        for step in cast(
            "tuple[MigrationStep, ...]",
            resolve_fully_qualified_name(hop.fully_qualified_name).migrations,
        )[hop.start : hop.stop]
        for change in step.result_changes
    )
    if not changes:
        return None
    spec_fields = link.current.fields
    return lambda manifest: _apply_result_changes(
        manifest, changes, spec_fields=spec_fields
    )


def result_source_for_loading(obj: Spec[Any]) -> _LoadableResult | None:
    if result_manifest_path_in(obj._base_dir).exists():
        return _LoadableResult(result_dir_in(obj._base_dir), None)
    link_path = result_link_path_in(obj._base_dir)
    if link_path.exists():
        link = _ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
        if not result_manifest_path_in(link.source.base_dir).exists():
            raise RuntimeError(f"{link_path} points to a missing result")
    else:
        link = _find_source(obj, _class_resolution(obj))
        if link is None:
            return None
        _write_result_link(obj, link)
    return _LoadableResult(
        result_dir_in(link.source.base_dir), _manifest_transform_for(link)
    )
