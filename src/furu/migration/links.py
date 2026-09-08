from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from furu.constants import FIELDSMARKER
from furu.locking import is_active_lock, lock, read_text_or_none
from furu.metadata import ArtifactSpec
from furu.migration.resolution import (
    _apply_child_moves,
    _apply_steps,
    _class_resolution,
    _ClassResolution,
    _Covered,
)
from furu.migration.steps import _describe_step
from furu.storage._layout import (
    compute_lock_path_in,
    metadata_path_in,
    result_dir_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.utils import JsonFields, _stable_json_dump, atomic_write_text

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


def _read_link(artifact_dir: Path) -> _ResultLink | None:
    if (link_text := read_text_or_none(result_link_path_in(artifact_dir))) is None:
        return None
    link = _ResultLink.model_validate_json(link_text)
    return link if result_manifest_path_in(link.source.base_dir).exists() else None


def _read_source(artifact_dir: Path) -> _ResultLink | None:
    """What the artifact directory is or points at, done or still computing.

    Whether the result is usable is checked live by the caller (manifest for
    done, compute lock for running), so the index stays valid as jobs finish.
    """
    if (link := _read_link(artifact_dir)) is not None:
        return link
    if (metadata_text := read_text_or_none(metadata_path_in(artifact_dir))) is None:
        return None
    artifact = ArtifactSpec.model_validate(json.loads(metadata_text)["artifact"])
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


_SOURCES_CACHE: dict[tuple[type, Path], Mapping[str, list[_ResultLink]]] = {}


def _migrated_key(
    resolution: _ClassResolution, covered: _Covered, fields: JsonFields
) -> str:
    if covered.child_moves:
        fields = {
            name: _apply_child_moves(value, covered.child_moves)
            for name, value in fields.items()
        }
    return _stable_json_dump(
        _apply_steps(resolution.own, covered.generation.start, fields)
    )


def _migrated_sources(
    cls: type, resolution: _ClassResolution, covered: _Covered
) -> Mapping[str, list[_ResultLink]]:
    key = (cls, covered.schema_directory)
    if (sources := _SOURCES_CACHE.get(key)) is None:
        sources = {}
        if covered.schema_directory.exists():
            for artifact_dir in sorted(covered.schema_directory.iterdir()):
                if not artifact_dir.is_dir():
                    continue
                if (source_link := _read_source(artifact_dir)) is None:
                    continue
                sources.setdefault(
                    _migrated_key(resolution, covered, source_link.current.fields), []
                ).append(source_link)
        _SOURCES_CACHE[key] = sources
    return sources


def _pinned_matches(covered: _Covered, target_fields: JsonFields) -> bool:
    return all(
        # Serialized so NaN defaults compare equal, like the index keys.
        _stable_json_dump(target_fields[name]) == _stable_json_dump(value)
        for name, value in covered.generation.pinned.items()
    )


def _sources(
    obj: Spec, resolution: _ClassResolution
) -> Iterator[tuple[_Covered, _ResultLink]]:
    """Artifacts under covered schemas whose migrated fields equal obj's."""
    if not resolution.covered:
        return
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    target_key = _stable_json_dump(target_fields)
    for covered in resolution.covered:
        if not _pinned_matches(covered, target_fields):
            continue
        sources = _migrated_sources(type(obj), resolution, covered)
        for source_link in sources.get(target_key, ()):
            yield covered, source_link


def _find_source(obj: Spec, resolution: _ClassResolution) -> _ResultLink | None:
    for covered, source_link in _sources(obj, resolution):
        if not result_manifest_path_in(source_link.source.base_dir).exists():
            continue
        return _ResultLink(
            current=_ResultLinkCurrent(
                fully_qualified_name=obj._fully_qualified_name,
                schema_hash=obj._artifact_schema_hash,
                artifact_hash=obj._artifact_hash,
                fields=cast(JsonFields, obj._artifact_data[FIELDSMARKER]),
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
        )
    return None


def _is_running_elsewhere(obj: Spec, resolution: _ClassResolution) -> bool:
    """Whether a job under an older schema is computing obj's result right now."""
    return any(
        is_active_lock(compute_lock_path_in(source_link.source.base_dir))
        for _, source_link in _sources(obj, resolution)
    )


def migrates_to(artifact: ArtifactSpec, obj: Spec) -> bool:
    """Whether a result stored for ``artifact`` would be obj's after migration."""
    resolution = _class_resolution(obj)
    schema_directory = (
        obj._metadata.storage
        / Path(*artifact.fully_qualified_name.split("."))
        / artifact.schema_hash
    )
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    for covered in resolution.covered:
        if covered.schema_directory != schema_directory:
            continue
        return _pinned_matches(covered, target_fields) and _migrated_key(
            resolution, covered, cast(JsonFields, artifact.artifact_data[FIELDSMARKER])
        ) == _stable_json_dump(target_fields)
    return False


def result_dir_for_loading(obj: Spec, *, has_lock: bool = False) -> Path | None:
    if result_manifest_path_in(obj._base_dir).exists():
        return result_dir_in(obj._base_dir)
    if link := _read_link(obj._base_dir):
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
