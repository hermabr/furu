from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from furu.core import (
    _internal_furu_dir_in,
    _metadata_path_in,
    _result_dir_in,
    _result_manifest_path_in,
)
from furu.utils import JsonValue, fully_qualified_name, nfs_safe_unique_name

if TYPE_CHECKING:
    from furu.core import Furu


type JsonFields = dict[str, JsonValue]


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


@dataclass(frozen=True, slots=True)
class _Source:
    ultimate_data_dir: Path
    ultimate_fully_qualified_name: str
    ultimate_schema_hash: str
    ultimate_artifact_hash: str
    fields: JsonFields
    prior_migration_path: tuple[MigrationStep, ...]


_Node = tuple[str, str]


def _result_link_path_in(data_dir: Path) -> Path:
    return _internal_furu_dir_in(data_dir) / "result-link.json"


def result_dir_for_loading(obj: Furu[Any]) -> Path | None:
    if obj._result_manifest_path.exists():
        return obj._result_dir
    link_path = _result_link_path_in(obj.data_dir)
    if not link_path.exists():
        return None
    link = json.loads(link_path.read_text(encoding="utf-8"))
    return _result_dir_in(Path(link["source"]["data_dir"]))


def _migration_paths_ending_at(
    target_node: _Node,
    migrations: tuple[Migration, ...],
) -> list[tuple[Migration, ...]]:
    by_target: dict[_Node, list[Migration]] = {}
    for migration in migrations:
        new_node = (migration.new_fully_qualified_name, migration.new_schema_hash)
        by_target.setdefault(new_node, []).append(migration)

    paths: list[tuple[Migration, ...]] = []

    def visit(
        node: _Node,
        suffix: tuple[Migration, ...],
        visited: frozenset[_Node],
    ) -> None:
        for migration in by_target.get(node, ()):
            old_node = (migration.old_fully_qualified_name, migration.old_schema_hash)
            if old_node in visited:
                continue
            new_path = (migration, *suffix)
            paths.append(new_path)
            visit(old_node, new_path, visited | {old_node})

    visit(target_node, (), frozenset({target_node}))
    return paths


def _source_from_artifact_dir(artifact_dir: Path) -> _Source | None:
    result_manifest = _result_manifest_path_in(artifact_dir)
    metadata_path = _metadata_path_in(artifact_dir)

    if result_manifest.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact = metadata["artifact"]
        data = artifact["data"]
        return _Source(
            ultimate_data_dir=artifact_dir,
            ultimate_fully_qualified_name=data["|class"],
            ultimate_schema_hash=artifact["schema_hash"],
            ultimate_artifact_hash=artifact["hash"],
            fields=data["fields"],
            prior_migration_path=(),
        )

    link_path = _result_link_path_in(artifact_dir)
    if link_path.exists():
        link = json.loads(link_path.read_text(encoding="utf-8"))
        current = link["current"]
        source = link["source"]
        return _Source(
            ultimate_data_dir=Path(source["data_dir"]),
            ultimate_fully_qualified_name=source["fully_qualified_name"],
            ultimate_schema_hash=source["schema_hash"],
            ultimate_artifact_hash=source["artifact_hash"],
            fields=current["fields"],
            prior_migration_path=tuple(
                MigrationStep(**step) for step in link["migration_path"]
            ),
        )

    return None


def _iter_candidate_artifacts(
    obj: Furu[Any],
    node: _Node,
) -> Iterator[_Source]:
    fqn, schema_hash = node
    schema_dir = obj.storage_root / Path(*fqn.split(".")) / schema_hash
    if not schema_dir.exists():
        return

    for artifact_dir in schema_dir.iterdir():
        if not artifact_dir.is_dir():
            continue
        source = _source_from_artifact_dir(artifact_dir)
        if source is not None:
            yield source


def _find_source_for_path(
    obj: Furu[Any],
    target_fields: JsonFields,
    migration_path: tuple[Migration, ...],
) -> _Source | None:
    first = migration_path[0]
    source_node = (first.old_fully_qualified_name, first.old_schema_hash)

    for candidate in _iter_candidate_artifacts(obj, source_node):
        fields = candidate.fields
        for migration in migration_path:
            fields = migration.transform_fn(fields)
        if fields == target_fields:
            return candidate
    return None


def _write_result_link(
    obj: Furu[Any],
    source: _Source,
    migration_path: tuple[MigrationStep, ...],
) -> None:
    obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
    artifact_data = cast(dict[str, JsonValue], obj.artifact_data)
    link = {
        "kind": "result_link",
        "current": {
            "fully_qualified_name": fully_qualified_name(type(obj)),
            "schema_hash": obj.artifact_schema_hash,
            "artifact_hash": obj.artifact_hash,
            "fields": artifact_data["fields"],
        },
        "source": {
            "fully_qualified_name": source.ultimate_fully_qualified_name,
            "schema_hash": source.ultimate_schema_hash,
            "artifact_hash": source.ultimate_artifact_hash,
            "data_dir": str(source.ultimate_data_dir),
        },
        "migration_path": [asdict(step) for step in migration_path],
    }
    link_path = _result_link_path_in(obj.data_dir)
    tmp = nfs_safe_unique_name(link_path, name="tmp")
    tmp.write_text(json.dumps(link, indent=2), encoding="utf-8")
    tmp.rename(link_path)


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

    for migration_path in _migration_paths_ending_at(target_node, migrations):
        source = _find_source_for_path(
            obj=obj,
            target_fields=target_fields,
            migration_path=migration_path,
        )
        if source is not None:
            full_path = source.prior_migration_path + tuple(
                MigrationStep.from_migration(m) for m in migration_path
            )
            _write_result_link(obj=obj, source=source, migration_path=full_path)
            return True
    return False
