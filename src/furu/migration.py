from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from furu.result import MANIFEST_FILE_NAME
from furu.utils import JsonValue, fully_qualified_name, nfs_safe_unique_name

if TYPE_CHECKING:
    from furu.core import Furu

JsonFields: TypeAlias = dict[str, JsonValue]
MigrationNode: TypeAlias = tuple[str, str]

RESULT_LINK_FILENAME = "result-link.json"


@dataclass(frozen=True, slots=True)
class Migration:
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str
    transform_fn: Callable[[JsonFields], JsonFields]


@dataclass(frozen=True, slots=True)
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
class Source:
    ultimate_source_data_dir: Path
    fully_qualified_name: str
    schema_hash: str
    artifact_hash: str
    fields: JsonFields
    prior_migration_path: tuple[MigrationStep, ...]


def result_link_path(obj: Furu[Any]) -> Path:
    return obj._internal_furu_dir / RESULT_LINK_FILENAME


def read_result_link(obj: Furu[Any]) -> dict[str, Any] | None:
    link_path = result_link_path(obj)
    if not link_path.exists():
        return None
    return cast(dict[str, Any], json.loads(link_path.read_text(encoding="utf-8")))


def result_dir_for_loading(obj: Furu[Any]) -> Path | None:
    if obj._result_manifest_path.exists():
        return obj._result_dir

    link = read_result_link(obj)
    if link is None:
        return None

    return Path(link["source"]["data_dir"]) / "result"


def migrate(obj: Furu[Any]) -> bool:
    if result_link_path(obj).exists():
        return True

    if obj._result_manifest_path.exists():
        return False

    target_node = (fully_qualified_name(type(obj)), obj.artifact_schema_hash)
    target_fields = cast(JsonFields, cast(dict[str, Any], obj.artifact_data)["fields"])

    for migration_path in migration_paths_ending_at(
        target_node, type(obj).migrations()
    ):
        source = find_source_for_path(
            obj=obj,
            target_fields=target_fields,
            migration_path=migration_path,
        )
        if source is not None:
            write_result_link(
                obj=obj,
                source=source,
                migration_path=source.prior_migration_path
                + tuple(MigrationStep.from_migration(m) for m in migration_path),
            )
            return True

    return False


def migration_paths_ending_at(
    target_node: MigrationNode,
    migrations: tuple[Migration, ...],
) -> list[tuple[Migration, ...]]:
    incoming: dict[MigrationNode, list[Migration]] = {}
    for migration in migrations:
        incoming.setdefault(
            (migration.new_fully_qualified_name, migration.new_schema_hash), []
        ).append(migration)

    paths: list[tuple[Migration, ...]] = []

    def visit(node: MigrationNode, suffix: tuple[Migration, ...]) -> None:
        for migration in incoming.get(node, []):
            next_suffix = (migration, *suffix)
            paths.append(next_suffix)
            visit(
                (migration.old_fully_qualified_name, migration.old_schema_hash),
                next_suffix,
            )

    visit(target_node, ())
    return paths


def find_source_for_path(
    *,
    obj: Furu[Any],
    target_fields: JsonFields,
    migration_path: tuple[Migration, ...],
) -> Source | None:
    first = migration_path[0]
    source_node = (first.old_fully_qualified_name, first.old_schema_hash)

    for candidate in iter_candidate_artifacts(obj, source_node):
        fields = candidate.fields
        for migration in migration_path:
            fields = migration.transform_fn(fields)
        if fields == target_fields:
            return candidate

    return None


def iter_candidate_artifacts(
    obj: Furu[Any],
    node: MigrationNode,
) -> Iterator[Source]:
    fully_qualified_name_, schema_hash = node
    schema_dir = (
        obj.storage_root / Path(*fully_qualified_name_.split(".")) / schema_hash
    )

    if not schema_dir.exists():
        return

    for artifact_dir in schema_dir.iterdir():
        if not artifact_dir.is_dir():
            continue
        source = source_from_artifact_dir(artifact_dir)
        if source is not None:
            yield source


def source_from_artifact_dir(artifact_dir: Path) -> Source | None:
    if (artifact_dir / "result" / MANIFEST_FILE_NAME).exists():
        metadata = _read_metadata_artifact(artifact_dir)
        artifact_data = cast(dict[str, Any], metadata["data"])
        return Source(
            ultimate_source_data_dir=artifact_dir,
            fully_qualified_name=artifact_data["|class"],
            schema_hash=metadata["schema_hash"],
            artifact_hash=metadata["hash"],
            fields=cast(JsonFields, artifact_data["fields"]),
            prior_migration_path=(),
        )

    link_path = artifact_dir / ".furu" / RESULT_LINK_FILENAME
    if link_path.exists():
        link = cast(dict[str, Any], json.loads(link_path.read_text(encoding="utf-8")))
        current = link["current"]
        source = link["source"]

        return Source(
            ultimate_source_data_dir=Path(source["data_dir"]),
            fully_qualified_name=current["fully_qualified_name"],
            schema_hash=current["schema_hash"],
            artifact_hash=current["artifact_hash"],
            fields=cast(JsonFields, current["fields"]),
            prior_migration_path=tuple(
                MigrationStep(
                    old_fully_qualified_name=step["old_fully_qualified_name"],
                    old_schema_hash=step["old_schema_hash"],
                    new_fully_qualified_name=step["new_fully_qualified_name"],
                    new_schema_hash=step["new_schema_hash"],
                )
                for step in link["migration_path"]
            ),
        )

    return None


def write_result_link(
    *,
    obj: Furu[Any],
    source: Source,
    migration_path: tuple[MigrationStep, ...],
) -> None:
    obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)

    link = {
        "kind": "result_link",
        "current": {
            "fully_qualified_name": fully_qualified_name(type(obj)),
            "schema_hash": obj.artifact_schema_hash,
            "artifact_hash": obj.artifact_hash,
            "fields": cast(dict[str, Any], obj.artifact_data)["fields"],
        },
        "source": {
            "fully_qualified_name": source.fully_qualified_name,
            "schema_hash": source.schema_hash,
            "artifact_hash": source.artifact_hash,
            "data_dir": str(source.ultimate_source_data_dir),
        },
        "migration_path": [
            {
                "old_fully_qualified_name": step.old_fully_qualified_name,
                "old_schema_hash": step.old_schema_hash,
                "new_fully_qualified_name": step.new_fully_qualified_name,
                "new_schema_hash": step.new_schema_hash,
            }
            for step in migration_path
        ],
    }

    write_json_atomically(result_link_path(obj), link)


def write_json_atomically(path: Path, value: Any) -> None:
    tmp_path = nfs_safe_unique_name(path, name="tmp")
    tmp_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp_path.rename(path)


def _read_metadata_artifact(artifact_dir: Path) -> dict[str, Any]:
    metadata_path = artifact_dir / ".furu" / "metadata.json"
    metadata = cast(
        dict[str, Any], json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    return cast(dict[str, Any], metadata["artifact"])
