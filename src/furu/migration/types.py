from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

from pydantic import JsonValue

from furu.metadata import CompletedMetadata


@dataclass(frozen=True, slots=True)
class Migration:
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str
    transform_fn: Callable[
        [dict[str, JsonValue]],
        dict[str, JsonValue],
    ]


@dataclass(frozen=True, slots=True)
class ResolvedMigration:
    source_metadata: CompletedMetadata
    migration_path: tuple[Migration, ...]


type MigrationNode = tuple[str, str]


@dataclass(frozen=True, slots=True)
class _MigrationEdge:
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str


@dataclass(frozen=True, slots=True)
class MigrationEdgeIdentity:
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str

    @classmethod
    def from_migration(cls, migration: Migration | _MigrationEdge) -> Self:
        return cls(
            old_fully_qualified_name=migration.old_fully_qualified_name,
            old_schema_hash=migration.old_schema_hash,
            new_fully_qualified_name=migration.new_fully_qualified_name,
            new_schema_hash=migration.new_schema_hash,
        )
