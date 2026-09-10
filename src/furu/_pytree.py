"""Minimal pytree walking for trees whose leaves are Specs.

A Spec is always a leaf: flattening never descends into a Spec's own fields
(those are dependencies, discovered by the DAG, not part of the tree's shape).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from furu.core import Spec


def children(value: object) -> dict[Any, object] | None:
    """Keyed children of a container, or None if ``value`` is not a container."""
    match value:
        case list() | tuple() | set() | frozenset():
            return dict(enumerate(value))
        case dict():
            return dict(value)
        case BaseModel():
            return {name: getattr(value, name) for name in type(value).model_fields}
        case _ if is_dataclass(value) and not isinstance(value, type):
            return {f.name: getattr(value, f.name) for f in fields(value)}
    return None


def _rebuild(value: object, new_children: dict[Any, object]) -> object:
    """Return a copy of container ``value`` with its children replaced."""
    match value:
        case list():
            return list(new_children.values())
        case tuple() if (make := getattr(type(value), "_make", None)) is not None:
            return make(new_children.values())  # NamedTuple
        case tuple():
            return tuple(new_children.values())
        case dict():
            return dict(new_children)
        case BaseModel():
            return value.model_copy(update=new_children)
        case _:
            out = copy.copy(value)
            for name, child in new_children.items():
                object.__setattr__(out, name, child)
            return out


def flatten_specs(tree: object) -> tuple[list[Spec], Callable[[Sequence[Any]], Any]]:
    """Split a pytree of Specs into its leaves and a function rebuilding its shape."""
    from furu.core import Spec

    leaves: list[Spec] = []

    def collect(value: object, path: str) -> None:
        if isinstance(value, Spec):
            leaves.append(value)
        elif isinstance(value, set | frozenset):
            raise TypeError(f"sets are not supported as Spec containers at {path}")
        elif (kids := children(value)) is None:
            raise TypeError(
                f"expected Spec leaves, got {type(value).__qualname__} at {path}"
            )
        else:
            for key, child in kids.items():
                collect(child, f"{path}[{key!r}]")

    def unflatten(values: Sequence[Any]) -> Any:
        it = iter(values)

        def fill(value: object) -> object:
            if isinstance(value, Spec):
                return next(it)
            kids = children(value)
            assert kids is not None
            return _rebuild(value, {key: fill(child) for key, child in kids.items()})

        return fill(tree)

    collect(tree, "tree")
    return leaves, unflatten
