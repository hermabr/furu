from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from furu.core import (
    _internal_furu_dir_in,
    _metadata_path_in,
    _result_dir_in,
    _result_manifest_path_in,
)
from furu.metadata import CompletedMetadata
from furu.utils import JsonFields, JsonValue, fully_qualified_name

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
    data_dir: Path


class _ResultLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    current: _ResultLinkCurrent
    source: _ResultLinkSource
    migration_path: tuple[MigrationStep, ...]


_Node = tuple[str, str]


def _result_link_path_in(data_dir: Path) -> Path:
    return _internal_furu_dir_in(data_dir) / "result-link.json"


def result_dir_for_loading(obj: Furu[Any]) -> Path | None:
    if obj._result_manifest_path.exists():
        return obj._result_dir
    link_path = _result_link_path_in(obj.data_dir)
    if not link_path.exists():
        return None
    link = _ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
    return _result_dir_in(link.source.data_dir)


def migrate(obj: Furu[Any]) -> bool:
    if _result_link_path_in(obj.data_dir).exists():
        return True
    if obj._result_manifest_path.exists():
        return False

    migrations = type(obj).migrations()
    if not migrations:
        return False

    target_node = (
        fully_qualified_name(type(obj)),
        obj.artifact_schema_hash,
    )
    artifact_data = cast(dict[str, JsonValue], obj.artifact_data)
    target_fields = cast(JsonFields, artifact_data["fields"])

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

            result_manifest = _result_manifest_path_in(artifact_dir)
            metadata_path = _metadata_path_in(artifact_dir)
            source_link: _ResultLink | None = None
            if result_manifest.exists() and metadata_path.exists():
                metadata = CompletedMetadata.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
                artifact = metadata.artifact
                data = cast(dict[str, JsonValue], artifact.data)
                artifact_fully_qualified_name = cast(str, data["|class"])
                artifact_fields = cast(JsonFields, data["fields"])
                source_link = _ResultLink(
                    current=_ResultLinkCurrent(
                        fully_qualified_name=artifact_fully_qualified_name,
                        schema_hash=artifact.schema_hash,
                        artifact_hash=artifact.hash,
                        fields=artifact_fields,
                    ),
                    source=_ResultLinkSource(
                        fully_qualified_name=artifact_fully_qualified_name,
                        schema_hash=artifact.schema_hash,
                        artifact_hash=artifact.hash,
                        data_dir=artifact_dir,
                    ),
                    migration_path=(),
                )
            else:
                link_path = _result_link_path_in(artifact_dir)
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
            obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
            _result_link_path_in(obj.data_dir).write_text(
                result_link.model_dump_json(indent=2), encoding="utf-8"
            )
            return True
    return False
