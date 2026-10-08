from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from furu.core import Spec

type _Container = (
    list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any] | dict[Any, Any]
)


def _as_container(tree: object) -> _Container | None:
    """The one definition of which values furu looks inside for specs.

    Dataclasses and pydantic models are viewed as dicts of their fields, so
    create() returns them as plain dicts instead of objects whose fields no
    longer match their annotations.
    """
    from furu.core import Spec

    match tree:
        case Spec():
            return None
        case list() | tuple() | set() | frozenset() | dict():
            return tree
        case PydanticBaseModel():
            return {name: getattr(tree, name) for name in type(tree).model_fields}
        case _ if is_dataclass(tree) and not isinstance(tree, type):
            return {field.name: getattr(tree, field.name) for field in fields(tree)}
    return None


def specs_in(tree: object) -> Iterator[Spec]:
    from furu.core import Spec

    if isinstance(tree, Spec):
        yield tree
        return
    match _as_container(tree):
        case dict() as children:
            for child in children.values():
                yield from specs_in(child)
        case None:
            pass
        case children:
            for child in children:
                yield from specs_in(child)


def map_specs(fn: Callable[[Spec], object], tree: object) -> Any:
    """Replace every spec in `tree` with `fn(spec)`, returning plain containers."""
    from furu.core import Spec

    if isinstance(tree, Spec):
        return fn(tree)
    match _as_container(tree):
        case None:
            return tree
        case dict() as children:
            return {key: map_specs(fn, child) for key, child in children.items()}
        case list() as children:
            return [map_specs(fn, child) for child in children]
        case tuple() as children:
            return tuple(map_specs(fn, child) for child in children)
        case set() as children:
            return {map_specs(fn, child) for child in children}
        case children:
            return frozenset(map_specs(fn, child) for child in children)
