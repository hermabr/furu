from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from furu.core import Furu
from furu.dependencies import collect_declared_refs
from furu.metadata import ArtifactSpec
from furu.migration import result_dir_for_loading


class NodeKey(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    object_id: str
    data_path: str


class ArtifactNode(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    key: NodeKey
    artifact: ArtifactSpec


class GraphEdge(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    dependency: NodeKey
    dependent: NodeKey


class GraphFragment(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    nodes: tuple[ArtifactNode, ...]
    edges: tuple[GraphEdge, ...]
    roots: tuple[NodeKey, ...]
    done: tuple[NodeKey, ...]


def _canonical_data_path(path: Path) -> str:
    return str(path.resolve(strict=False))


def node_key_for(obj: Furu[Any]) -> NodeKey:
    return NodeKey(
        object_id=obj.object_id,
        data_path=_canonical_data_path(obj.data_dir),
    )


def artifact_spec_for(obj: Furu[Any]) -> ArtifactSpec:
    return ArtifactSpec(
        fully_qualified_name=obj._fully_qualified_name,
        data=obj.artifact_data,
        artifact_hash=obj.artifact_hash,
        schema=obj.schema,
        schema_hash=obj.artifact_schema_hash,
    )


def artifact_node_for(obj: Furu[Any]) -> ArtifactNode:
    return ArtifactNode(
        key=node_key_for(obj),
        artifact=artifact_spec_for(obj),
    )


def _key_sort_value(key: NodeKey) -> tuple[str, str]:
    return (key.data_path, key.object_id)


def _edge_sort_value(edge: GraphEdge) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_key_sort_value(edge.dependency), _key_sort_value(edge.dependent))


def discover_missing_closure(
    roots: Sequence[Furu[Any]],
) -> GraphFragment:
    queue = deque(roots)
    seen: set[NodeKey] = set()
    nodes: dict[NodeKey, ArtifactNode] = {}
    edges: set[GraphEdge] = set()
    done: set[NodeKey] = set()

    while queue:
        obj = queue.popleft()
        key = node_key_for(obj)

        if key in seen:
            continue

        seen.add(key)
        nodes.setdefault(key, artifact_node_for(obj))

        if result_dir_for_loading(obj) is not None:
            done.add(key)
            continue

        for ref in collect_declared_refs(obj):
            ref_key = node_key_for(ref)
            nodes.setdefault(ref_key, artifact_node_for(ref))
            edges.add(
                GraphEdge(
                    dependency=ref_key,
                    dependent=key,
                )
            )
            queue.append(ref)

    root_keys = tuple(node_key_for(root) for root in roots)
    return GraphFragment(
        nodes=tuple(
            node
            for _, node in sorted(
                nodes.items(), key=lambda item: _key_sort_value(item[0])
            )
        ),
        edges=tuple(sorted(edges, key=_edge_sort_value)),
        roots=root_keys,
        done=tuple(sorted(done, key=_key_sort_value)),
    )
