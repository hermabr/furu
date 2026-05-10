from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict

from furu.metadata import CompletedMetadata
from furu.utils import JsonValue, class_label

if TYPE_CHECKING:
    from furu.core import Furu

_HEX_HASH_PATTERN: Final = re.compile(r"^[0-9a-f]+$")
_RESULT_LINK_FILE_NAME: Final = "result-link.json"
_MANIFEST_REL: Final = Path("result", "manifest.json")
_METADATA_REL: Final = Path(".furu", "metadata.json")


class DuplicateMigrationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Migration:
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str
    transform_fn: Callable[[dict[str, JsonValue]], dict[str, JsonValue]]

    @property
    def old_node(self) -> _Node:
        return (self.old_fully_qualified_name, self.old_schema_hash)

    @property
    def new_node(self) -> _Node:
        return (self.new_fully_qualified_name, self.new_schema_hash)

    @property
    def edge_id(self) -> tuple[str, str, str, str]:
        return (
            self.old_fully_qualified_name,
            self.old_schema_hash,
            self.new_fully_qualified_name,
            self.new_schema_hash,
        )


type _Node = tuple[str, str]
type _Graph = dict[_Node, list[Migration]]


@dataclass(frozen=True, slots=True)
class _ResolvedMigration:
    source_fully_qualified_name: str
    source_schema_hash: str
    source_artifact_hash: str
    source_data_dir: Path
    migration_path: tuple[Migration, ...]


class _MigrationEdgeRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    old_fully_qualified_name: str
    old_schema_hash: str
    new_fully_qualified_name: str
    new_schema_hash: str


class _CurrentNodeRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    fully_qualified_name: str
    schema_hash: str
    artifact_hash: str


class _SourceNodeRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    fully_qualified_name: str
    schema_hash: str
    artifact_hash: str
    data_dir: Path


class ResultLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["result_link"] = "result_link"
    current: _CurrentNodeRef
    source: _SourceNodeRef
    migration_path: tuple[_MigrationEdgeRef, ...]


def _validate_fqn(value: object, *, field: str, cls_label_str: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Migration on {cls_label_str}: {field} must be a non-empty string, "
            f"got {value!r}"
        )


def _validate_hash(value: object, *, field: str, cls_label_str: str) -> None:
    if not isinstance(value, str) or not _HEX_HASH_PATTERN.match(value):
        raise ValueError(
            f"Migration on {cls_label_str}: {field} must be a raw lowercase hex "
            f"hash, got {value!r}"
        )


def _validate_migration_fields(migration: Migration, *, cls_label_str: str) -> None:
    _validate_fqn(
        migration.old_fully_qualified_name,
        field="old_fully_qualified_name",
        cls_label_str=cls_label_str,
    )
    _validate_fqn(
        migration.new_fully_qualified_name,
        field="new_fully_qualified_name",
        cls_label_str=cls_label_str,
    )
    _validate_hash(
        migration.old_schema_hash,
        field="old_schema_hash",
        cls_label_str=cls_label_str,
    )
    _validate_hash(
        migration.new_schema_hash,
        field="new_schema_hash",
        cls_label_str=cls_label_str,
    )
    if not callable(migration.transform_fn):
        raise TypeError(
            f"Migration on {cls_label_str}: transform_fn must be callable, got "
            f"{type(migration.transform_fn).__name__}"
        )


def _validate_class_migrations(
    migrations: object,
    *,
    cls_label_str: str,
) -> tuple[Migration, ...]:
    if not isinstance(migrations, tuple):
        raise TypeError(
            f"{cls_label_str}.migrations() must return a tuple, got "
            f"{type(migrations).__name__}"
        )
    seen: set[tuple[str, str, str, str]] = set()
    validated: list[Migration] = []
    for migration in migrations:
        if not isinstance(migration, Migration):
            raise TypeError(
                f"{cls_label_str}.migrations() must return Migration instances, got "
                f"{type(migration).__name__}"
            )
        _validate_migration_fields(migration, cls_label_str=cls_label_str)
        if migration.edge_id in seen:
            raise DuplicateMigrationError(
                f"{cls_label_str}.migrations() declares duplicate migration from "
                f"({migration.old_fully_qualified_name!r}, "
                f"{migration.old_schema_hash!r}) to "
                f"({migration.new_fully_qualified_name!r}, "
                f"{migration.new_schema_hash!r})"
            )
        seen.add(migration.edge_id)
        validated.append(migration)
    return tuple(validated)


def _build_graph(migrations: Iterable[Migration]) -> _Graph:
    graph: _Graph = {}
    for migration in migrations:
        graph.setdefault(migration.old_node, []).append(migration)
        graph.setdefault(migration.new_node, [])
    return graph


def _find_paths(
    graph: _Graph,
    start: _Node,
    end: _Node,
) -> list[tuple[Migration, ...]]:
    if start == end:
        return [()]
    if start not in graph or end not in graph:
        return []

    results: list[tuple[Migration, ...]] = []
    visited: set[_Node] = {start}
    path: list[Migration] = []

    def dfs(node: _Node) -> None:
        for migration in graph.get(node, ()):
            next_node = migration.new_node
            path.append(migration)
            if next_node == end:
                results.append(tuple(path))
            elif next_node not in visited:
                visited.add(next_node)
                dfs(next_node)
                visited.discard(next_node)
            path.pop()

    dfs(start)
    return results


def _reachable_predecessors(graph: _Graph, end: _Node) -> set[_Node]:
    reverse: dict[_Node, list[_Node]] = {}
    for node, edges in graph.items():
        for migration in edges:
            reverse.setdefault(migration.new_node, []).append(node)

    visited: set[_Node] = {end}
    queue: list[_Node] = [end]
    while queue:
        node = queue.pop()
        for prev in reverse.get(node, ()):
            if prev not in visited:
                visited.add(prev)
                queue.append(prev)
    visited.discard(end)
    return visited


def _apply_path(
    fields: dict[str, JsonValue],
    path: Iterable[Migration],
) -> dict[str, JsonValue]:
    from furu.core import Furu

    current = fields
    for migration in path:
        result = migration.transform_fn(current)
        if isinstance(result, Furu):
            raise TypeError(
                f"Migration transform_fn from "
                f"({migration.old_fully_qualified_name!r}, "
                f"{migration.old_schema_hash!r}) to "
                f"({migration.new_fully_qualified_name!r}, "
                f"{migration.new_schema_hash!r}) returned a Furu object; "
                "transform_fn must return dict[str, JsonValue]"
            )
        if not isinstance(result, dict):
            raise TypeError(
                f"Migration transform_fn from "
                f"({migration.old_fully_qualified_name!r}, "
                f"{migration.old_schema_hash!r}) to "
                f"({migration.new_fully_qualified_name!r}, "
                f"{migration.new_schema_hash!r}) must return dict[str, JsonValue], "
                f"got {type(result).__name__}"
            )
        current = result
    return current


def _completed_old_artifacts(
    storage_root: Path,
    *,
    old_node: _Node,
) -> Iterator[tuple[Path, CompletedMetadata]]:
    fqn, schema_hash = old_node
    schema_dir = storage_root.joinpath(*fqn.split("."), schema_hash)
    if not schema_dir.is_dir():
        return
    for artifact_dir in schema_dir.iterdir():
        if not artifact_dir.is_dir():
            continue
        if not (artifact_dir / _MANIFEST_REL).exists():
            continue
        metadata_path = artifact_dir / _METADATA_REL
        if not metadata_path.exists():
            continue
        try:
            metadata = CompletedMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        yield artifact_dir, metadata


def _extract_fields(data: JsonValue) -> dict[str, JsonValue] | None:
    if not isinstance(data, dict):
        return None
    fields = data.get("fields")
    if not isinstance(fields, dict):
        return None
    return fields


def _find_migrated_result(obj: Furu) -> _ResolvedMigration | None:
    migrations = type(obj).migrations()
    if not migrations:
        return None

    graph = _build_graph(migrations)
    current_node: _Node = (obj._fully_qualified_name, obj.artifact_schema_hash)
    if current_node not in graph:
        return None

    requested_fields = _extract_fields(obj.artifact_data)
    if requested_fields is None:
        return None

    predecessors = _reachable_predecessors(graph, current_node)
    if not predecessors:
        return None

    storage_root = obj.storage_root
    logger = obj.logger

    for old_node in predecessors:
        paths = _find_paths(graph, old_node, current_node)
        if not paths:
            continue
        for source_data_dir, source_metadata in _completed_old_artifacts(
            storage_root, old_node=old_node
        ):
            old_fields = _extract_fields(source_metadata.artifact.data)
            if old_fields is None:
                continue
            for path in paths:
                try:
                    transformed = _apply_path(old_fields, path)
                except Exception:
                    logger.warning(
                        "migration transform raised while resolving %s from %s",
                        class_label(type(obj)),
                        source_data_dir,
                        exc_info=True,
                    )
                    continue
                if transformed == requested_fields:
                    return _ResolvedMigration(
                        source_fully_qualified_name=old_node[0],
                        source_schema_hash=source_metadata.artifact.schema_hash,
                        source_artifact_hash=source_metadata.artifact.hash,
                        source_data_dir=source_data_dir,
                        migration_path=path,
                    )
    return None


def _result_link_path_for(internal_furu_dir: Path) -> Path:
    return internal_furu_dir / _RESULT_LINK_FILE_NAME


def _write_result_link(obj: Furu, resolved: _ResolvedMigration) -> None:
    link = ResultLink(
        current=_CurrentNodeRef(
            fully_qualified_name=obj._fully_qualified_name,
            schema_hash=obj.artifact_schema_hash,
            artifact_hash=obj.artifact_hash,
        ),
        source=_SourceNodeRef(
            fully_qualified_name=resolved.source_fully_qualified_name,
            schema_hash=resolved.source_schema_hash,
            artifact_hash=resolved.source_artifact_hash,
            data_dir=resolved.source_data_dir,
        ),
        migration_path=tuple(
            _MigrationEdgeRef(
                old_fully_qualified_name=m.old_fully_qualified_name,
                old_schema_hash=m.old_schema_hash,
                new_fully_qualified_name=m.new_fully_qualified_name,
                new_schema_hash=m.new_schema_hash,
            )
            for m in resolved.migration_path
        ),
    )
    obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
    _result_link_path_for(obj._internal_furu_dir).write_text(
        link.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _read_and_verify_result_link(obj: Furu) -> _ResolvedMigration | None:
    link_path = _result_link_path_for(obj._internal_furu_dir)
    if not link_path.exists():
        return None
    try:
        link = ResultLink.model_validate_json(link_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if (
        link.current.fully_qualified_name != obj._fully_qualified_name
        or link.current.schema_hash != obj.artifact_schema_hash
        or link.current.artifact_hash != obj.artifact_hash
    ):
        return None

    source_data_dir = link.source.data_dir
    if not (source_data_dir / _MANIFEST_REL).exists():
        return None

    source_metadata_path = source_data_dir / _METADATA_REL
    if not source_metadata_path.exists():
        return None
    try:
        source_metadata = CompletedMetadata.model_validate_json(
            source_metadata_path.read_text(encoding="utf-8")
        )
    except Exception:
        return None
    if (
        source_metadata.artifact.schema_hash != link.source.schema_hash
        or source_metadata.artifact.hash != link.source.artifact_hash
    ):
        return None

    by_id = {m.edge_id: m for m in type(obj).migrations()}
    resolved_path: list[Migration] = []
    for edge in link.migration_path:
        edge_id = (
            edge.old_fully_qualified_name,
            edge.old_schema_hash,
            edge.new_fully_qualified_name,
            edge.new_schema_hash,
        )
        migration = by_id.get(edge_id)
        if migration is None:
            return None
        resolved_path.append(migration)

    if not resolved_path:
        return None
    if (
        resolved_path[0].old_fully_qualified_name != link.source.fully_qualified_name
        or resolved_path[0].old_schema_hash != link.source.schema_hash
        or resolved_path[-1].new_fully_qualified_name != obj._fully_qualified_name
        or resolved_path[-1].new_schema_hash != obj.artifact_schema_hash
    ):
        return None

    old_fields = _extract_fields(source_metadata.artifact.data)
    requested_fields = _extract_fields(obj.artifact_data)
    if old_fields is None or requested_fields is None:
        return None

    try:
        transformed = _apply_path(old_fields, resolved_path)
    except Exception:
        return None
    if transformed != requested_fields:
        return None

    return _ResolvedMigration(
        source_fully_qualified_name=link.source.fully_qualified_name,
        source_schema_hash=link.source.schema_hash,
        source_artifact_hash=link.source.artifact_hash,
        source_data_dir=source_data_dir,
        migration_path=tuple(resolved_path),
    )


def resolve_migrated_result_dir(obj: Furu) -> Path | None:
    """Return the result dir satisfying ``obj`` via a migration, or ``None``.

    Tries an existing result-link marker first; falls back to migration
    discovery and writes a marker on a fresh hit.
    """
    resolved = _read_and_verify_result_link(obj)
    if resolved is None:
        resolved = _find_migrated_result(obj)
        if resolved is None:
            return None
        _write_result_link(obj, resolved)
    return resolved.source_data_dir / "result"
