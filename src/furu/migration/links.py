from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from furu.constants import FIELDSMARKER
from furu.metadata import CompletedMetadata
from furu.migration.field_values import _added_default_fields, _apply_steps
from furu.migration.scanner import _class_resolution, _ClassResolution
from furu.migration.steps import _describe_step
from furu.storage._layout import (
    metadata_path_in,
    result_dir_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.utils import JsonFields, nfs_safe_unique_name

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
    if not resolution.covered:
        return None
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    added_defaults = _added_default_fields(obj, resolution)
    for generation, schema_directory in resolution.covered:
        if not schema_directory.exists():
            continue
        for artifact_dir in sorted(schema_directory.iterdir()):
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
    # A linked result still marks this schema directory as holding a result.
    _record_schema_snapshot(obj)


def result_dir_for_loading(obj: Spec[Any]) -> Path | None:
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
