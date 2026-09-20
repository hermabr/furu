from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from furu._declared_types import declared_result_type
from furu.constants import FIELDSMARKER
from furu.locking import lock, read_text_or_none
from furu.metadata import CompletedMetadata
from furu.migration.resolution import (
    _apply_child_moves,
    _apply_steps,
    _class_resolution,
    _ClassResolution,
    _Covered,
)
from furu.migration.steps import MigrationError, _describe_step
from furu.result.bundle import load_result_bundle
from furu.storage._layout import (
    compute_lock_path_in,
    data_dir_in,
    metadata_path_in,
    result_dir_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.utils import (
    JsonFields,
    _stable_json_dump,
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


_SOURCES_CACHE: dict[tuple[type, Path], Mapping[str, list[_ResultLink]]] = {}


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
                fields = source_link.current.fields
                if covered.child_moves:
                    fields = {
                        name: _apply_child_moves(value, covered.child_moves)
                        for name, value in fields.items()
                    }
                fields = _apply_steps(resolution.own, covered.generation.start, fields)
                sources.setdefault(_stable_json_dump(fields), []).append(source_link)
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
        for source_link in sources.get(target_key, ()):
            if not result_manifest_path_in(source_link.source.base_dir).exists():
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
            )
    return None


def _source_dir_for_loading(obj: Spec, *, has_lock: bool = False) -> Path | None:
    """The base directory holding obj's result: its own when it has a manifest,
    otherwise the source of its recorded link, or of a freshly found source in
    the migration chain (recording the link). None when there is no result."""
    if result_manifest_path_in(obj._base_dir).exists():
        return obj._base_dir
    if link := _read_source(obj._base_dir):
        return link.source.base_dir
    link = _find_source(obj, _class_resolution(obj))
    if link is None:
        return None

    obj._base_dir.mkdir(parents=True, exist_ok=True)
    if not has_lock:
        with lock(compute_lock_path_in(obj._base_dir)):
            return _source_dir_for_loading(obj, has_lock=True)

    from furu.execution.load_or_create import _record_schema_snapshot

    atomic_write_text(
        result_link_path_in(obj._base_dir), link.model_dump_json(indent=2)
    )
    _record_schema_snapshot(obj)
    return link.source.base_dir


def _covering(
    cls: type, resolution: _ClassResolution, source_dir: Path
) -> _Covered | None:
    """The covered generation through which ``source_dir`` is reachable:
    its own schema directory, or one holding a link that points at it."""
    for covered in resolution.covered:
        if covered.schema_directory == source_dir.parent:
            return covered
    for covered in resolution.covered:
        for links in _migrated_sources(cls, resolution, covered).values():
            if any(link.source.base_dir == source_dir for link in links):
                return covered
    return None


def _rewrites_for(obj: Spec, source_dir: Path) -> tuple[Callable[[Any], Any], ...]:
    """The result_rewrite callables between the source's generation and now."""
    cls = type(obj)
    steps = cls.migrations
    if not any(step.result_rewrite is not None for step in steps):
        return ()
    resolution = _class_resolution(obj)
    covered = _covering(cls, resolution, source_dir)
    if covered is None:
        raise MigrationError(
            f"{obj._log_label} links to {source_dir}, whose schema is no longer "
            "covered by the migration chain, so its result_rewrite steps cannot "
            "be replayed"
        )
    if covered.schema_directory != source_dir.parent:
        # Reached through another class's link. That class's own rewrites are
        # not part of this chain and cannot be replayed from here.
        via = resolve_fully_qualified_name(covered.generation.class_name)
        if any(
            step.result_rewrite is not None for step in getattr(via, "migrations", ())
        ):
            raise MigrationError(
                f"{obj._log_label} reaches {source_dir} through a link recorded "
                f"by {covered.generation.class_name}, whose migrations carry "
                "result_rewrite steps that this chain cannot replay; declare the "
                f"full chain from the source on {cls.__name__}.migrations"
            )
    return tuple(
        step.result_rewrite
        for step in steps[covered.generation.start :]
        if step.result_rewrite is not None
    )


def load_result[T](obj: Spec[T], *, has_lock: bool = False) -> T | None:
    """obj's stored result, or None when there is none yet.

    Results reached through a link are replayed through the chain's
    ``result_rewrite`` callables so they match the current ``create()`` shape.
    """
    source_dir = _source_dir_for_loading(obj, has_lock=has_lock)
    if source_dir is None:
        return None
    rewrites = _rewrites_for(obj, source_dir) if source_dir != obj._base_dir else ()
    return cast(
        T,
        load_result_bundle(
            result_dir_in(source_dir),
            data_dir=data_dir_in(source_dir),
            declared_type=declared_result_type(type(obj)),
            rewrites=rewrites,
        ),
    )
