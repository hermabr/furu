from __future__ import annotations

from .ast import AndNode, NotNode, OrNode, Query

_MAX_QUERY_NODES = 200
_MAX_QUERY_DEPTH = 30


def _child_nodes(node: Query) -> tuple[Query, ...]:
    if isinstance(node, AndNode | OrNode):
        return tuple(node.args)
    if isinstance(node, NotNode):
        return (node.arg,)
    return ()


def validate_query(
    query: Query,
    *,
    max_node_count: int = _MAX_QUERY_NODES,
    max_depth: int = _MAX_QUERY_DEPTH,
) -> None:
    node_count = 0
    stack: list[tuple[Query, int]] = [(query, 1)]
    while stack:
        current, depth = stack.pop()
        node_count += 1

        if node_count > max_node_count:
            raise ValueError(f"query exceeds max node count ({max_node_count})")
        if depth > max_depth:
            raise ValueError(f"query exceeds max depth ({max_depth})")

        for child in _child_nodes(current):
            stack.append((child, depth + 1))
