from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel

type IsLeaf = Callable[[object], bool]
type NodeKind = Literal[
    "list", "tuple", "set", "frozenset", "dict", "dataclass", "pydantic"
]


@dataclass(frozen=True)
class PyTreeNode:
    kind: NodeKind
    value: object
    entries: tuple[object, ...]
    children: tuple[object, ...]

    def unflatten(self, children: Iterable[object]) -> object:
        values = tuple(children)
        match self.kind:
            case "list" | "set" | "frozenset":
                return type(self.value)(values)
            case "tuple":
                return (
                    type(self.value)(*values)
                    if hasattr(type(self.value), "_fields")
                    else type(self.value)(values)
                )
            case "dict":
                return type(self.value)(zip(self.entries, values, strict=True))
            case "dataclass":
                result = copy.copy(self.value)
                for name, child in zip(self.entries, values, strict=True):
                    object.__setattr__(result, cast(str, name), child)
                return result
            case "pydantic":
                assert isinstance(self.value, BaseModel)
                return self.value.model_copy(
                    update=dict(
                        zip(cast(tuple[str, ...], self.entries), values, strict=True)
                    )
                )


def tree_node(value: object) -> PyTreeNode | None:
    if isinstance(value, BaseModel):
        entries = tuple(type(value).model_fields)
        kind: NodeKind = "pydantic"
        children = tuple(getattr(value, name) for name in entries)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        entries = tuple(field.name for field in dataclasses.fields(value))
        kind = "dataclass"
        children = tuple(getattr(value, name) for name in entries)
    elif isinstance(value, (list, tuple)):
        children = tuple(value)
        entries = tuple(range(len(children)))
        kind = "list" if isinstance(value, list) else "tuple"
    elif isinstance(value, (set, frozenset)):
        children = tuple(
            sorted(
                value,
                key=lambda item: (
                    type(item).__module__, type(item).__qualname__, repr(item)
                ),
            )
        )
        entries = tuple(range(len(children)))
        kind = "frozenset" if isinstance(value, frozenset) else "set"
    elif isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        entries = tuple(mapping)
        children = tuple(mapping[key] for key in entries)
        kind = "dict"
    else:
        return None
    return PyTreeNode(kind, value, entries, children)


def tree_map(
    function: Callable[[object], object],
    value: object,
    *,
    is_leaf: IsLeaf | None = None,
) -> object:
    node = None if is_leaf is not None and is_leaf(value) else tree_node(value)
    if node is None:
        return function(value)
    return node.unflatten(
        tree_map(function, child, is_leaf=is_leaf) for child in node.children
    )


def tree_leaves(value: object, *, is_leaf: IsLeaf | None = None) -> list[object]:
    leaves: list[object] = []

    def collect(leaf: object) -> object:
        leaves.append(leaf)
        return leaf

    tree_map(collect, value, is_leaf=is_leaf)
    return leaves
