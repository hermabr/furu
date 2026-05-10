from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue, TypeAdapter, ValidationError

from furu.constants import CLASSMARKER
from furu.metadata import CompletedMetadata, Metadata
from furu.migration.result_link import (
    _ResultLink,
    _ResultLinkArtifact,
    _ResultLinkSource,
)
from furu.migration.types import (
    Migration,
    MigrationEdgeIdentity,
    MigrationNode,
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

    source_fully_qualified_name = _artifact_fully_qualified_name(source_metadata)
    if source_fully_qualified_name is None:
        return None

    if link.source != _ResultLinkSource(
        fully_qualified_name=source_fully_qualified_name,
        schema_hash=source_metadata.artifact.schema_hash,
        artifact_hash=source_metadata.artifact.hash,
        data_dir=str(source_metadata.data_path),
    ):
        return None

    path = _resolve_registered_path(type(obj), link.migration_path)
    if path is None:
        return None

    fields = _artifact_fields(source_metadata)
    if fields is None:
        return None
    migrated_fields = _apply_migration_path(path, fields)
    if migrated_fields != _requested_fields(obj):
        return None

    return ResolvedMigration(
        source_metadata=source_metadata,
        migration_path=path,
    )


def find_migrated_result(obj: Furu[Any]) -> ResolvedMigration | None:
    current_node = (obj._fully_qualified_name, obj.artifact_schema_hash)
    requested_fields = _requested_fields(obj)
    graph = _MigrationGraph(_registered_migrations_for_class(type(obj)))

    for old_metadata in _completed_old_artifacts(obj.storage_root):
        old_fully_qualified_name = _artifact_fully_qualified_name(old_metadata)
        old_fields = _artifact_fields(old_metadata)
        if old_fully_qualified_name is None or old_fields is None:
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
    source_fully_qualified_name = _artifact_fully_qualified_name(match.source_metadata)
    if source_fully_qualified_name is None:
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


class _MigrationGraph:
    def __init__(self, migrations: Iterable[Migration]) -> None:
        self._by_old: dict[MigrationNode, list[Migration]] = {}
        for migration in migrations:
            self._by_old.setdefault(
                (migration.old_fully_qualified_name, migration.old_schema_hash),
                [],
            ).append(migration)

    def paths(
        self, start: MigrationNode, end: MigrationNode
    ) -> Iterable[tuple[Migration, ...]]:
        yield from self._paths(start, end, visited={start}, path=())

    def _paths(
        self,
        current: MigrationNode,
        end: MigrationNode,
        *,
        visited: set[MigrationNode],
        path: tuple[Migration, ...],
    ) -> Iterable[tuple[Migration, ...]]:
        if current == end:
            yield path
            return

        for migration in self._by_old.get(current, ()):
            next_node = (
                migration.new_fully_qualified_name,
                migration.new_schema_hash,
            )
            if next_node in visited:
                continue
            yield from self._paths(
                next_node,
                end,
                visited=visited | {next_node},
                path=(*path, migration),
            )


def _completed_old_artifacts(storage_root: Path) -> Iterable[CompletedMetadata]:
    if not storage_root.exists():
        return

    for metadata_path in storage_root.glob("**/.furu/metadata.json"):
        metadata = _read_completed_metadata(metadata_path)
        if (
            metadata is not None
            and (metadata.data_path / "result" / "manifest.json").exists()
        ):
            yield metadata


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


def _artifact_fully_qualified_name(metadata: CompletedMetadata) -> str | None:
    value = metadata.artifact.data.get(CLASSMARKER)
    return value if isinstance(value, str) else None


def _artifact_fields(metadata: CompletedMetadata) -> dict[str, JsonValue] | None:
    fields = metadata.artifact.data.get("fields")
    if not isinstance(fields, dict):
        return None
    try:
        return TypeAdapter(dict[str, JsonValue]).validate_python(fields)
    except ValidationError:
        return None


def _requested_fields(obj: Furu[Any]) -> dict[str, JsonValue]:
    fields = obj.artifact_data.get("fields")
    if not isinstance(fields, dict):
        raise TypeError("Furu artifact_data must contain a fields object")
    return TypeAdapter(dict[str, JsonValue]).validate_python(fields)


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


def _resolve_registered_path(
    cls: type[Furu[Any]], raw_path: list[_MigrationEdge]
) -> tuple[Migration, ...] | None:
    by_identity = {
        MigrationEdgeIdentity.from_migration(migration): migration
        for migration in _registered_migrations_for_class(cls)
    }
    path: list[Migration] = []
    for edge in raw_path:
        migration = by_identity.get(MigrationEdgeIdentity.from_migration(edge))
        if migration is None:
            return None
        path.append(migration)
    return tuple(path)
