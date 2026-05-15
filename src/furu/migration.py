from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from furu.constants import FIELDSMARKER
from furu._storage_layout import (
    metadata_path_in,
    result_dir_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.metadata import CompletedMetadata
from furu.utils import JsonFields, fully_qualified_name

if TYPE_CHECKING:
    from furu.core import Furu


@dataclass(frozen=True, slots=True, kw_only=True)
class Migration:
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str
    transform_fn: Callable[[JsonFields], JsonFields]


@dataclass(frozen=True, slots=True, kw_only=True)
class MigrationStep:
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str

    @classmethod
    def from_migration(cls, migration: Migration) -> MigrationStep:
        return cls(
            old_fully_qualified_name=migration.old_fully_qualified_name,
            old_schema_hash=migration.old_schema_hash,
            new_fully_qualified_name=migration.new_fully_qualified_name,
            new_schema_hash=migration.new_schema_hash,
        )


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
    migration_path: tuple[MigrationStep, ...]


_Node = tuple[str, str]


def result_dir_for_loading[T](obj: Furu[T]) -> Path | None:
    if result_manifest_path_in(obj._base_dir).exists():
        return result_dir_in(obj._base_dir)
    link_path = result_link_path_in(obj._base_dir)
    if not link_path.exists():
        return None
    link = _ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
    result_dir = result_dir_in(link.source.base_dir)
    if not result_manifest_path_in(link.source.base_dir).exists():
        raise RuntimeError(f"{link_path} points to a missing result")
    return result_dir


def migrate[T](obj: Furu[T]) -> bool:
    if result_link_path_in(obj._base_dir).exists():
        result_dir_for_loading(obj)
        return True
    if result_manifest_path_in(obj._base_dir).exists():
        return False

    migrations = type(obj).migrations()
    if not migrations:
        return False

    target_node = (
        fully_qualified_name(type(obj)),
        obj.artifact_schema_hash,
    )
    target_fields = cast(JsonFields, obj.artifact_data[FIELDSMARKER])

    by_target: defaultdict[_Node, list[Migration]] = defaultdict(list)
    for migration in migrations:
        new_node = (migration.new_fully_qualified_name, migration.new_schema_hash)
        by_target[new_node].append(migration)

    def visit(
        node: _Node,
        suffix: tuple[Migration, ...],
        visited: frozenset[_Node],
    ) -> Iterator[tuple[Migration, ...]]:
        for migration in by_target.get(node, ()):
            old_node = (migration.old_fully_qualified_name, migration.old_schema_hash)
            if old_node in visited:
                continue
            new_path = (migration, *suffix)
            yield new_path
            yield from visit(old_node, new_path, visited | {old_node})

    for migration_path in visit(target_node, (), frozenset({target_node})):
        first = migration_path[0]
        schema_dir = (
            obj.storage_root
            / Path(*first.old_fully_qualified_name.split("."))
            / first.old_schema_hash
        )
        if not schema_dir.exists():
            continue

        for artifact_dir in schema_dir.iterdir():
            if not artifact_dir.is_dir():
                continue

            result_manifest = result_manifest_path_in(artifact_dir)
            metadata_path = metadata_path_in(artifact_dir)
            source_link: _ResultLink | None = None
            if result_manifest.exists() and metadata_path.exists():
                metadata = CompletedMetadata.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
                artifact = metadata.artifact
                artifact_fields = cast(JsonFields, artifact.artifact_data[FIELDSMARKER])
                source_link = _ResultLink(
                    current=_ResultLinkCurrent(
                        fully_qualified_name=artifact.fully_qualified_name,
                        schema_hash=artifact.schema_hash,
                        artifact_hash=artifact.artifact_hash,
                        fields=artifact_fields,
                    ),
                    source=_ResultLinkSource(
                        fully_qualified_name=artifact.fully_qualified_name,
                        schema_hash=artifact.schema_hash,
                        artifact_hash=artifact.artifact_hash,
                        base_dir=artifact_dir,
                    ),
                    migration_path=(),
                )
            else:
                link_path = result_link_path_in(artifact_dir)
                if link_path.exists():
                    source_link = _ResultLink.model_validate_json(
                        link_path.read_text(encoding="utf-8")
                    )

            if source_link is None:
                continue

            fields = source_link.current.fields
            for step in migration_path:
                fields = step.transform_fn(fields)
            if fields != target_fields:
                continue

            full_path = source_link.migration_path + tuple(
                MigrationStep.from_migration(step) for step in migration_path
            )
            result_link = _ResultLink(
                current=_ResultLinkCurrent(
                    fully_qualified_name=fully_qualified_name(type(obj)),
                    schema_hash=obj.artifact_schema_hash,
                    artifact_hash=obj.artifact_hash,
                    fields=target_fields,
                ),
                source=source_link.source,
                migration_path=full_path,
            )
            obj._base_dir.mkdir(parents=True, exist_ok=True)
            result_link_path_in(obj._base_dir).write_text(
                result_link.model_dump_json(indent=2), encoding="utf-8"
            )
            return True
    return False
