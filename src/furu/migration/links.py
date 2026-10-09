from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from furu._declared_types import declared_result_type
from furu.code_trace import is_valid, valid_version
from furu.constants import FIELDSMARKER
from furu.locking import lock, read_text_or_none
from furu.metadata import ArtifactSpec
from furu.migration.resolution import (
    _apply_child_moves,
    _apply_steps,
    _class_resolution,
    _ClassResolution,
    _Covered,
)
from furu.migration.steps import MigrationStep, _describe_step
from furu.result.bundle import load_result_bundle
from furu.storage._layout import (
    compute_lock_path_in,
    data_dir_in,
    result_dir_in,
    result_link_path_in,
    spec_path_in,
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
    version_dir: Path


class _ResultLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    current: _ResultLinkCurrent
    source: _ResultLinkSource
    migration_path: tuple[str, ...]


def _covered_in(
    resolution: _ClassResolution, schema_directory: Path
) -> _Covered | None:
    return next(
        (
            covered
            for covered in resolution.covered
            if covered.schema_directory == schema_directory
        ),
        None,
    )


def _read_link(obj: Spec) -> _ResultLink | None:
    if (link_text := read_text_or_none(result_link_path_in(obj._base_dir))) is None:
        return None
    link = _ResultLink.model_validate_json(link_text)
    if not is_valid(link.source.version_dir, storage_root=obj._metadata.storage):
        return None
    schema_directory = link.source.version_dir.parent.parent
    if _covered_in(_class_resolution(obj), schema_directory) is None:
        return None
    return link


_SOURCES_CACHE: dict[
    tuple[type, Path], Mapping[str, list[tuple[ArtifactSpec, Path]]]
] = {}


def _migrated_sources(
    cls: type, resolution: _ClassResolution, covered: _Covered
) -> Mapping[str, list[tuple[ArtifactSpec, Path]]]:
    """Source identities in a covered schema directory, keyed by migrated fields."""
    key = (cls, covered.schema_directory)
    if (sources := _SOURCES_CACHE.get(key)) is None:
        sources = {}
        if covered.schema_directory.exists():
            for identity_dir in sorted(covered.schema_directory.iterdir()):
                if not identity_dir.is_dir():
                    continue
                spec_path = spec_path_in(identity_dir)
                if not spec_path.exists():
                    continue
                artifact = ArtifactSpec.model_validate_json(
                    spec_path.read_text(encoding="utf-8")
                )
                source = (artifact, identity_dir)
                fields = cast(JsonFields, artifact.artifact_data[FIELDSMARKER])
                if covered.child_moves:
                    fields = {
                        name: _apply_child_moves(value, covered.child_moves)
                        for name, value in fields.items()
                    }
                fields = _apply_steps(resolution.own, covered.generation.start, fields)
                sources.setdefault(_stable_json_dump(fields), []).append(source)
        _SOURCES_CACHE[key] = sources
    return sources


def _find_source(obj: Spec, resolution: _ClassResolution) -> _ResultLink | None:
    if not resolution.covered:
        return None
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    target_key = _stable_json_dump(target_fields)
    for covered in resolution.covered:
        if any(
            # Serialized so NaN defaults compare equal, like the key below.
            _stable_json_dump(target_fields[name]) != _stable_json_dump(value)
            for name, value in covered.generation.pinned.items()
        ):
            continue
        sources = _migrated_sources(type(obj), resolution, covered)
        for artifact, identity_dir in sources.get(target_key, ()):
            version = valid_version(
                identity_dir, None, storage_root=obj._metadata.storage
            )
            if version is None:
                continue
            return _ResultLink(
                current=_ResultLinkCurrent(
                    fully_qualified_name=obj._fully_qualified_name,
                    schema_hash=obj._artifact_schema_hash,
                    artifact_hash=obj._artifact_hash,
                    fields=target_fields,
                ),
                source=_ResultLinkSource(
                    fully_qualified_name=artifact.fully_qualified_name,
                    schema_hash=artifact.schema_hash,
                    artifact_hash=artifact.artifact_hash,
                    version_dir=version,
                ),
                migration_path=tuple(
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


def own_version(obj: Spec) -> Path | None:
    return valid_version(
        obj._base_dir,
        obj._metadata.code_version,
        storage_root=obj._metadata.storage,
    )


def version_for_loading(obj: Spec, *, has_lock: bool = False) -> Path | None:
    """The version directory to load obj from: its own, or a migration source."""
    if (version := own_version(obj)) is not None:
        return version
    if link := _read_link(obj):
        return link.source.version_dir
    link = _find_source(obj, _class_resolution(obj))
    if link is None:
        return None

    obj._base_dir.mkdir(parents=True, exist_ok=True)
    if not has_lock:
        with lock(compute_lock_path_in(obj._base_dir)):
            return version_for_loading(obj, has_lock=True)

    from furu.execution.load_or_create import _record_schema_snapshot

    atomic_write_text(
        result_link_path_in(obj._base_dir), link.model_dump_json(indent=2)
    )
    _record_schema_snapshot(obj)
    return link.source.version_dir


def load_stored_result[T](obj: Spec[T], version: Path) -> T:
    steps: tuple[MigrationStep, ...] = ()
    if version.parent != obj._base_dir:
        resolution = _class_resolution(obj)
        covered = _covered_in(resolution, version.parent.parent)
        # version_for_loading only hands out covered sources.
        assert covered is not None, f"{version} is not covered by {obj._log_label}"
        steps = resolution.own.steps[covered.generation.start :]
    value = load_result_bundle(
        result_dir_in(version),
        data_dir=data_dir_in(version),
        declared_type=declared_result_type(type(obj)),
        rewrites=[
            step.result_rewrite for step in steps if step.result_rewrite is not None
        ],
    )
    return cast(T, value)
