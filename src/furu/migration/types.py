from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

from pydantic import JsonValue

from furu.metadata import CompletedMetadata
from furu.schema import schema_type
from furu.utils import _hash_dict_deterministically, fully_qualified_name


@dataclass(frozen=True, slots=True)
class _MigrationEdge:
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

    @classmethod
    def from_classes(
        cls,
        old: type[Any],
        new: type[Any],
        transform_fn: Callable[
            [dict[str, JsonValue]],
            dict[str, JsonValue],
        ],
    ) -> Self:
        return cls(
            old_fully_qualified_name=fully_qualified_name(old),
            old_schema_hash=_hash_dict_deterministically(schema_type(old, set())),
            new_fully_qualified_name=fully_qualified_name(new),
            new_schema_hash=_hash_dict_deterministically(schema_type(new, set())),
            transform_fn=transform_fn,
        )


@dataclass(frozen=True, slots=True)
class ResolvedMigration:
    source_metadata: CompletedMetadata
    source_fully_qualified_name: str
    migration_path: tuple[Migration, ...]


type MigrationNode = tuple[str, str]
