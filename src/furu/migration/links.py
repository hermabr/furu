from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from furu._declared_types import declared_result_type
from furu.constants import FIELDSMARKER
from furu.metadata import CompletedMetadata
from furu.migration.resolution import (
    _apply_child_moves,
    _apply_steps,
    _class_resolution,
    _ClassResolution,
)
from furu.migration.results import _apply_result_migrations
from furu.migration.steps import MigrationError, MigrationStep, _describe_step
from furu.result.bundle import load_result_manifest
from furu.storage._layout import (
    data_dir_in,
    metadata_path_in,
    result_dir_in,
    result_link_path_in,
    result_manifest_overlay_path_in,
    result_manifest_path_in,
)
from furu.utils import JsonFields, JsonValue, atomic_write_text

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


class _ResultLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    current: _ResultLinkCurrent
    source: _ResultLinkSource
    migration_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ReadableSource:
    link: _ResultLink
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class _ResultMatch:
    link: _ResultLink
    source_manifest_path: Path
    result_steps: tuple[MigrationStep, ...]


def _read_source(artifact_dir: Path) -> _ReadableSource | None:
    result_manifest = result_manifest_path_in(artifact_dir)
    metadata_path = metadata_path_in(artifact_dir)
    if result_manifest.exists() and metadata_path.exists():
        metadata = CompletedMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        return _ReadableSource(
            link=_ResultLink(
                current=_ResultLinkCurrent(
                    fully_qualified_name=metadata.artifact.fully_qualified_name,
                    schema_hash=metadata.artifact.schema_hash,
                    artifact_hash=metadata.artifact.artifact_hash,
                    fields=cast(
                        JsonFields, metadata.artifact.artifact_data[FIELDSMARKER]
                    ),
                ),
                source=_ResultLinkSource(
                    fully_qualified_name=metadata.artifact.fully_qualified_name,
                    schema_hash=metadata.artifact.schema_hash,
                    artifact_hash=metadata.artifact.artifact_hash,
                    base_dir=artifact_dir,
                ),
                migration_path=(),
            ),
            manifest_path=result_manifest,
        )
    link_path = result_link_path_in(artifact_dir)
    if link_path.exists():
        link = _ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
        overlay_path = result_manifest_overlay_path_in(artifact_dir)
        return _ReadableSource(
            link=link,
            manifest_path=(
                overlay_path
                if overlay_path.exists()
                else result_manifest_path_in(link.source.base_dir)
            ),
        )
    return None


def _find_source(obj: Spec[Any], resolution: _ClassResolution) -> _ResultMatch | None:
    if not resolution.covered:
        return None
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    for covered in resolution.covered:
        if not covered.schema_directory.exists():
            continue
        for artifact_dir in sorted(covered.schema_directory.iterdir()):
            if not artifact_dir.is_dir():
                continue
            source = _read_source(artifact_dir)
            if source is None:
                continue
            fields = source.link.current.fields
            if covered.child_moves:
                fields = {
                    name: _apply_child_moves(value, covered.child_moves)
                    for name, value in fields.items()
                }
            fields = _apply_steps(resolution.own, covered.generation.start, fields)
            if fields != target_fields:
                continue
            return _ResultMatch(
                link=_ResultLink(
                    current=_ResultLinkCurrent(
                        fully_qualified_name=obj._fully_qualified_name,
                        schema_hash=obj._artifact_schema_hash,
                        artifact_hash=obj._artifact_hash,
                        fields=target_fields,
                    ),
                    source=source.link.source,
                    migration_path=source.link.migration_path
                    + tuple(
                        f"{move.chain.label}: {_describe_step(step)}"
                        for move in covered.child_moves.values()
                        for step in move.chain.steps[move.start :]
                    )
                    + tuple(
                        _describe_step(step)
                        for step in resolution.own.steps[covered.generation.start :]
                    ),
                ),
                source_manifest_path=source.manifest_path,
                # Embedded child steps migrate the parent's artifact fields, not
                # the parent's cached result. Only this Spec owns this result.
                result_steps=resolution.own.steps[covered.generation.start :],
            )
    return None


def _write_result_link(obj: Spec[Any], match: _ResultMatch) -> None:
    from furu.execution.load_or_create import _record_schema_snapshot

    source_manifest_path = result_manifest_path_in(match.link.source.base_dir)
    if not source_manifest_path.exists():
        raise RuntimeError(
            f"cannot migrate result for {obj._log_label}: source result is missing "
            f"at {source_manifest_path}"
        )

    raw_manifest = cast(
        JsonValue,
        json.loads(match.source_manifest_path.read_text(encoding="utf-8")),
    )
    migrated_manifest = _apply_result_migrations(raw_manifest, match.result_steps)
    declared_type = declared_result_type(type(obj))
    try:
        load_result_manifest(
            migrated_manifest,
            bundle_dir=result_dir_in(match.link.source.base_dir),
            data_dir=data_dir_in(obj._base_dir),
            declared_type=declared_type,
        )
    except Exception as exc:
        raise MigrationError(
            f"migrated result for {obj._log_label} does not decode as the current "
            f"declared result type {declared_type!r}: {exc}"
        ) from exc

    obj._base_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        result_manifest_overlay_path_in(obj._base_dir),
        json.dumps(migrated_manifest, indent=2),
    )
    atomic_write_text(
        result_link_path_in(obj._base_dir), match.link.model_dump_json(indent=2)
    )
    _record_schema_snapshot(obj)


def result_manifest_for_loading(obj: Spec[Any], result_dir: Path) -> Path:
    own_manifest = result_manifest_path_in(obj._base_dir)
    if own_manifest.exists():
        return own_manifest
    overlay = result_manifest_overlay_path_in(obj._base_dir)
    if overlay.exists():
        return overlay
    return result_dir / "manifest.json"


def result_dir_for_loading(obj: Spec[Any]) -> Path | None:
    if result_manifest_path_in(obj._base_dir).exists():
        return result_dir_in(obj._base_dir)
    link_path = result_link_path_in(obj._base_dir)
    if link_path.exists():
        link = _ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
        if not result_manifest_path_in(link.source.base_dir).exists():
            raise RuntimeError(f"{link_path} points to a missing result")
        return result_dir_in(link.source.base_dir)
    link = _find_source(obj, _class_resolution(obj))
    if link is None:
        return None
    _write_result_link(obj, link)
    return result_dir_in(link.link.source.base_dir)
