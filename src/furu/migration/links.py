from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from furu.constants import FIELDSMARKER
from furu.locking import lock, read_text_or_none
from furu.metadata import CompletedMetadata
from furu.migration.resolution import (
    _apply_child_moves,
    _apply_steps,
    _class_resolution,
    _ClassResolution,
)
from furu.migration.steps import _describe_step
from furu.storage._layout import (
    compute_lock_path_in,
    metadata_path_in,
    result_dir_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.utils import JsonFields, _hash_dict_deterministically, atomic_write_text

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
        )
    link_path = result_link_path_in(artifact_dir)
    if (link_text := read_text_or_none(link_path)) is None:
        return None
    link = _ResultLink.model_validate_json(link_text)
    return link if result_manifest_path_in(link.source.base_dir).exists() else None


@dataclass(frozen=True, slots=True)
class _IndexedSource:
    source: _ResultLinkSource
    migration_path: tuple[str, ...]


# Every old-generation artifact of a class, keyed by a digest of the fields it
# migrates to. Scanned once per class per process, like the class resolution
# itself: a source that finishes afterwards goes unnoticed and is merely
# recomputed, while one deleted afterwards is caught on lookup.
_SOURCE_INDEX: dict[tuple[type, Path], Mapping[str, Sequence[_IndexedSource]]] = {}


def _index_sources(
    resolution: _ClassResolution,
) -> Mapping[str, Sequence[_IndexedSource]]:
    index: dict[str, list[_IndexedSource]] = {}
    for covered in resolution.covered:
        if not covered.schema_directory.exists():
            continue
        migration_path = tuple(
            f"{move.chain.label}: {_describe_step(step)}"
            for move in covered.child_moves.values()
            for step in move.chain.steps[move.start :]
        ) + tuple(
            _describe_step(step)
            for step in resolution.own.steps[covered.generation.start :]
        )
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
            index.setdefault(_hash_dict_deterministically(fields), []).append(
                _IndexedSource(
                    source=source_link.source,
                    migration_path=source_link.migration_path + migration_path,
                )
            )
    return index


def _find_source(obj: Spec, resolution: _ClassResolution) -> _ResultLink | None:
    if not resolution.covered:
        return None
    key = (type(obj), obj._metadata.storage)
    if (index := _SOURCE_INDEX.get(key)) is None:
        index = _SOURCE_INDEX[key] = _index_sources(resolution)
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    for candidate in index.get(_hash_dict_deterministically(target_fields), ()):
        if not result_manifest_path_in(candidate.source.base_dir).exists():
            continue
        return _ResultLink(
            current=_ResultLinkCurrent(
                fully_qualified_name=obj._fully_qualified_name,
                schema_hash=obj._artifact_schema_hash,
                artifact_hash=obj._artifact_hash,
                fields=target_fields,
            ),
            source=candidate.source,
            migration_path=candidate.migration_path,
        )
    return None


def result_dir_for_loading(obj: Spec, *, has_lock: bool = False) -> Path | None:
    if result_manifest_path_in(obj._base_dir).exists():
        return result_dir_in(obj._base_dir)
    if link := _read_source(obj._base_dir):
        return result_dir_in(link.source.base_dir)
    link = _find_source(obj, _class_resolution(obj))
    if link is None:
        return None

    obj._base_dir.mkdir(parents=True, exist_ok=True)
    if not has_lock:
        with lock(compute_lock_path_in(obj._base_dir)):
            return result_dir_for_loading(obj, has_lock=True)

    from furu.execution.load_or_create import _record_schema_snapshot

    atomic_write_text(
        result_link_path_in(obj._base_dir), link.model_dump_json(indent=2)
    )
    _record_schema_snapshot(obj)
    return result_dir_in(link.source.base_dir)
