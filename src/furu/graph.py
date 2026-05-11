from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from furu.core import Furu
from furu.dependencies import collect_declared_refs
from furu.metadata import ArtifactSpec, artifact_spec_for
from furu.migration import result_dir_for_loading


class NodeKey(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    object_id: str
    data_path: str

    def __hash__(self) -> int:
        return hash((self.object_id, self.data_path))


class ArtifactNode(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    key: NodeKey
    artifact: ArtifactSpec


class DiscoveredGraph(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    nodes: tuple[ArtifactNode, ...]
    edges: tuple[tuple[NodeKey, NodeKey], ...]
    roots: tuple[NodeKey, ...]


def canonical_data_path(path: Path) -> str:
    return str(path.resolve(strict=False))


def node_key_for(obj: Furu[Any]) -> NodeKey:
    return NodeKey(
        object_id=obj.object_id,
        data_path=canonical_data_path(obj.data_dir),
    )


def discover_missing_closure(
    roots: Sequence[Furu[Any]],
) -> DiscoveredGraph:
    queue: deque[Furu[Any]] = deque(roots)
    seen: set[NodeKey] = set()

    nodes: dict[NodeKey, ArtifactNode] = {}
    edges: set[tuple[NodeKey, NodeKey]] = set()
    root_keys = tuple(node_key_for(root) for root in roots)

    while queue:
        obj = queue.popleft()
        key = node_key_for(obj)

        if key in seen:
            continue

        seen.add(key)
        nodes[key] = ArtifactNode(
            key=key,
            artifact=artifact_spec_for(obj),
        )

        if result_dir_for_loading(obj) is not None:
            continue

        for ref in collect_declared_refs(obj):
            ref_key = node_key_for(ref)

            if ref_key not in nodes:
                nodes[ref_key] = ArtifactNode(
                    key=ref_key,
                    artifact=artifact_spec_for(ref),
                )

            edges.add((ref_key, key))
            queue.append(ref)

    return DiscoveredGraph(
        nodes=tuple(nodes.values()),
        edges=tuple(sorted(edges, key=repr)),
        roots=root_keys,
    )
