from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
type MigrationEdgeIdentity = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class _MigrationEdge:
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str


def _edge_identity(edge: Migration | _MigrationEdge) -> MigrationEdgeIdentity:
    return (
        edge.old_fully_qualified_name,
        edge.old_schema_hash,
        edge.new_fully_qualified_name,
        edge.new_schema_hash,
    )
