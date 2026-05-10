from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    MigrationEdgeIdentity,
    ResolvedMigration,
    _MigrationEdge,
)
from furu.utils import _hash_dict_deterministically

if TYPE_CHECKING:
    from furu.core import Furu


def _registered_migrations_for_class(cls: type[Furu[Any]]) -> tuple[Migration, ...]:
    migrations = cls.migrations()
    seen: set[MigrationEdgeIdentity] = set()
    for migration in migrations:
        identity = MigrationEdgeIdentity.from_migration(migration)
        if identity in seen:
            raise ValueError(
                f"{cls.__name__}.migrations() returned duplicate migration edge "
                f"{identity!r}"
            )
        seen.add(identity)
    return migrations


def _resolve_result_manifest_path(obj: Furu[Any]) -> Path | None:
    local_result_manifest_path = obj._result_dir / "manifest.json"
    if local_result_manifest_path.exists():
        return local_result_manifest_path

    linked = read_and_verify_result_link(obj)
    if linked is not None:
        return linked.source_metadata.data_path / "result" / "manifest.json"

    match = find_migrated_result(obj)
    if match is not None:
        write_result_link(obj, match)
        return match.source_metadata.data_path / "result" / "manifest.json"

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
    if not (source_data_dir / "result" / "manifest.json").exists():
        return None

    source_metadata = _read_completed_metadata(
        source_data_dir / ".furu" / "metadata.json"
    )
    if source_metadata is None:
        return None

    source_fully_qualified_name = source_metadata.artifact.data.get(CLASSMARKER)
    if not isinstance(source_fully_qualified_name, str):
        return None

    source_fields = source_metadata.artifact.data.get("fields")

    if link.source != _ResultLinkSource(
        fully_qualified_name=source_fully_qualified_name,
        schema_hash=source_metadata.artifact.schema_hash,
        artifact_hash=source_metadata.artifact.hash,
        data_dir=str(source_metadata.data_path),
    ):
        return None

    by_identity = {
        MigrationEdgeIdentity.from_migration(migration): migration
        for migration in _registered_migrations_for_class(type(obj))
    }
    path: list[Migration] = []
    for edge in link.migration_path:
        migration = by_identity.get(MigrationEdgeIdentity.from_migration(edge))
        if migration is None:
            return None
        path.append(migration)

    if not isinstance(source_fields, dict):
        return None
    try:
        source_fields = TypeAdapter(dict[str, JsonValue]).validate_python(source_fields)
    except ValidationError:
        return None

    requested_fields = obj.artifact_data.get("fields")
    if not isinstance(requested_fields, dict):
        raise TypeError("Furu artifact_data must contain a fields object")
    requested_fields = TypeAdapter(dict[str, JsonValue]).validate_python(
        requested_fields
    )

    migration_path = tuple(path)
    migrated_fields = _apply_migration_path(migration_path, source_fields)
    if migrated_fields != requested_fields:
        return None
    return ResolvedMigration(
        source_metadata=source_metadata,
        migration_path=migration_path,
    )


def find_migrated_result(obj: Furu[Any]) -> ResolvedMigration | None:
    current_node = (obj._fully_qualified_name, obj.artifact_schema_hash)
    requested_fields = obj.artifact_data.get("fields")
    if not isinstance(requested_fields, dict):
        raise TypeError("Furu artifact_data must contain a fields object")
    requested_fields = TypeAdapter(dict[str, JsonValue]).validate_python(
        requested_fields
    )
    graph = _MigrationGraph(_registered_migrations_for_class(type(obj)))

    if not obj.storage_root.exists():
        return None

    for metadata_path in obj.storage_root.glob("**/.furu/metadata.json"):
        old_metadata = _read_completed_metadata(metadata_path)
        if (
            old_metadata is None
            or not (old_metadata.data_path / "result" / "manifest.json").exists()
        ):
            continue

        old_fully_qualified_name = old_metadata.artifact.data.get(CLASSMARKER)
        old_fields = old_metadata.artifact.data.get("fields")
        if not isinstance(old_fully_qualified_name, str) or not isinstance(
            old_fields, dict
        ):
            continue
        try:
            old_fields = TypeAdapter(dict[str, JsonValue]).validate_python(old_fields)
        except ValidationError:
            continue

        old_node = (old_fully_qualified_name, old_metadata.artifact.schema_hash)
        for path in graph.paths(old_node, current_node):
            migrated_fields = _apply_migration_path(path, old_fields)
            if migrated_fields == requested_fields:
                return ResolvedMigration(
                    source_metadata=old_metadata,
                    migration_path=path,
                )
    return None


def write_result_link(obj: Furu[Any], match: ResolvedMigration) -> None:
    source_fully_qualified_name = match.source_metadata.artifact.data.get(CLASSMARKER)
    if not isinstance(source_fully_qualified_name, str):
        raise ValueError("source artifact metadata has no fully qualified name")

    obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
    link = _ResultLink(
        current=_ResultLinkArtifact(
            fully_qualified_name=obj._fully_qualified_name,
            schema_hash=obj.artifact_schema_hash,
            artifact_hash=obj.artifact_hash,
        ),
        source=_ResultLinkSource(
            fully_qualified_name=source_fully_qualified_name,
            schema_hash=match.source_metadata.artifact.schema_hash,
            artifact_hash=match.source_metadata.artifact.hash,
            data_dir=str(match.source_metadata.data_path),
        ),
        migration_path=[
            _MigrationEdge(
                old_fully_qualified_name=migration.old_fully_qualified_name,
                old_schema_hash=migration.old_schema_hash,
                new_fully_qualified_name=migration.new_fully_qualified_name,
                new_schema_hash=migration.new_schema_hash,
            )
            for migration in match.migration_path
        ],
    )
    obj._result_link_path.write_text(link.model_dump_json(indent=2))


def _read_completed_metadata(path: Path) -> CompletedMetadata | None:
    try:
        raw_text = path.read_text()
    except (FileNotFoundError, OSError):
        return None

    try:
        metadata = TypeAdapter(Metadata).validate_json(raw_text)
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
            current = TypeAdapter(dict[str, JsonValue]).validate_python(transformed)
        except ValidationError as exc:
            raise TypeError(
                "Migration.transform_fn must return dict[str, JsonValue]"
            ) from exc
    return current
