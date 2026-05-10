from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from furu.constants import CLASSMARKER
from furu.metadata import CompletedMetadata, Metadata
from furu.migration.graph import _MigrationGraph
from furu.migration.result_link import (
    _ResultLink,
    _ResultLinkArtifact,
    _ResultLinkSource,
)
from furu.migration.types import (
    Migration,
    MigrationNode,
    ResolvedMigration,
    _MigrationEdge,
)
from furu.result import MANIFEST_FILE_NAME
from furu.utils import _hash_dict_deterministically

if TYPE_CHECKING:
    from furu.core import Furu


_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])
_METADATA_ADAPTER: Final = TypeAdapter(Metadata)


def _result_manifest_in(data_dir: Path) -> Path:
    return data_dir / "result" / MANIFEST_FILE_NAME


def _registered_migrations_for_class(cls: type[Furu[Any]]) -> tuple[Migration, ...]:
    migrations = cls.migrations()
    seen: set[_MigrationEdge] = set()
    for migration in migrations:
        edge = _MigrationEdge.from_migration(migration)
        if edge in seen:
            raise ValueError(
                f"{cls.__name__}.migrations() returned duplicate migration edge "
                f"{edge!r}"
            )
        seen.add(edge)
    return migrations


def _resolve_result_manifest_path(obj: Furu[Any]) -> Path | None:
    local = _result_manifest_in(obj.data_dir)
    if local.exists():
        return local

    linked = read_and_verify_result_link(obj)
    if linked is not None:
        return _result_manifest_in(linked.source_metadata.data_path)

    if not _registered_migrations_for_class(type(obj)):
        return None

    match = find_migrated_result(obj)
    if match is not None:
        write_result_link(obj, match)
        return _result_manifest_in(match.source_metadata.data_path)

    return None


def read_and_verify_result_link(obj: Furu[Any]) -> ResolvedMigration | None:
    try:
        link = _ResultLink.model_validate_json(obj._result_link_path.read_text())
    except (FileNotFoundError, OSError, ValidationError):
        return None

    if link.current != _ResultLinkArtifact(
        fully_qualified_name=obj._fully_qualified_name,
        schema_hash=obj.artifact_schema_hash,
        artifact_hash=obj.artifact_hash,
    ):
        return None

    source_data_dir = Path(link.source.data_dir)
    if not _result_manifest_in(source_data_dir).exists():
        return None

    source_metadata = _read_completed_metadata(
        source_data_dir / ".furu" / "metadata.json"
    )
    if source_metadata is None:
        return None

    decoded = _decode_source_node(source_metadata)
    if decoded is None:
        return None
    source_fully_qualified_name, source_fields = decoded

    if link.source != _ResultLinkSource(
        fully_qualified_name=source_fully_qualified_name,
        schema_hash=source_metadata.artifact.schema_hash,
        artifact_hash=source_metadata.artifact.hash,
        data_dir=str(source_metadata.data_path),
    ):
        return None

    by_edge = {
        _MigrationEdge.from_migration(migration): migration
        for migration in _registered_migrations_for_class(type(obj))
    }
    path: list[Migration] = []
    for edge in link.migration_path:
        migration = by_edge.get(edge)
        if migration is None:
            return None
        path.append(migration)

    requested_fields = _requested_fields(obj)
    migration_path = tuple(path)
    if _apply_migration_path(migration_path, source_fields) != requested_fields:
        return None
    return ResolvedMigration(
        source_metadata=source_metadata,
        source_fully_qualified_name=source_fully_qualified_name,
        migration_path=migration_path,
    )


def find_migrated_result(obj: Furu[Any]) -> ResolvedMigration | None:
    current_node = (obj._fully_qualified_name, obj.artifact_schema_hash)
    requested_fields = _requested_fields(obj)
    graph = _MigrationGraph(_registered_migrations_for_class(type(obj)))

    if not obj.storage_root.exists():
        return None

    for metadata_path in obj.storage_root.glob("**/.furu/metadata.json"):
        old_metadata = _read_completed_metadata(metadata_path)
        if (
            old_metadata is None
            or not _result_manifest_in(old_metadata.data_path).exists()
        ):
            continue

        decoded = _decode_source_node(old_metadata)
        if decoded is None:
            continue
        old_fully_qualified_name, old_fields = decoded

        old_node: MigrationNode = (
            old_fully_qualified_name,
            old_metadata.artifact.schema_hash,
        )
        for path in graph.paths(old_node, current_node):
            if _apply_migration_path(path, old_fields) == requested_fields:
                return ResolvedMigration(
                    source_metadata=old_metadata,
                    source_fully_qualified_name=old_fully_qualified_name,
                    migration_path=path,
                )
    return None


def write_result_link(obj: Furu[Any], match: ResolvedMigration) -> None:
    obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
    link = _ResultLink(
        current=_ResultLinkArtifact(
            fully_qualified_name=obj._fully_qualified_name,
            schema_hash=obj.artifact_schema_hash,
            artifact_hash=obj.artifact_hash,
        ),
        source=_ResultLinkSource(
            fully_qualified_name=match.source_fully_qualified_name,
            schema_hash=match.source_metadata.artifact.schema_hash,
            artifact_hash=match.source_metadata.artifact.hash,
            data_dir=str(match.source_metadata.data_path),
        ),
        migration_path=[
            _MigrationEdge.from_migration(migration)
            for migration in match.migration_path
        ],
    )
    obj._result_link_path.write_text(link.model_dump_json(indent=2))


def _decode_source_node(
    metadata: CompletedMetadata,
) -> tuple[str, dict[str, JsonValue]] | None:
    fully_qualified_name = metadata.artifact.data.get(CLASSMARKER)
    fields = metadata.artifact.data.get("fields")
    if not isinstance(fully_qualified_name, str) or not isinstance(fields, dict):
        return None
    return fully_qualified_name, fields


def _requested_fields(obj: Furu[Any]) -> dict[str, JsonValue]:
    fields = obj.artifact_data.get("fields")
    if not isinstance(fields, dict):
        raise TypeError("Furu artifact_data must contain a fields object")
    return fields


def _read_completed_metadata(path: Path) -> CompletedMetadata | None:
    try:
        raw_text = path.read_text()
    except (FileNotFoundError, OSError):
        return None

    try:
        metadata = _METADATA_ADAPTER.validate_json(raw_text)
    except ValidationError:
        return None
    if not isinstance(metadata, CompletedMetadata):
        return None
    if _hash_dict_deterministically(metadata.artifact.data) != metadata.artifact.hash:
        return None
    return metadata


def _apply_migration_path(
    path: tuple[Migration, ...], fields: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    current = fields
    for migration in path:
        transformed = migration.transform_fn(current)
        try:
            current = _FIELDS_ADAPTER.validate_python(transformed)
        except ValidationError as exc:
            raise TypeError(
                "Migration.transform_fn must return dict[str, JsonValue]"
            ) from exc
    return current
