from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from furu.core import Spec


def map_specs(fn: Callable[[Spec], object], tree: object) -> Any:
    """Replace every spec in `tree` with `fn(spec)`, returning plain containers.

    This is the one definition of which values furu looks inside for specs.
    Dataclasses and pydantic models come back as dicts of their fields rather
    than objects whose fields no longer match their annotations.
    """
    from furu.core import Spec

    match tree:
        case Spec():
            return fn(tree)
        case dict():
            return {key: map_specs(fn, child) for key, child in tree.items()}
        case list():
            return [map_specs(fn, child) for child in tree]
        case tuple():
            return tuple(map_specs(fn, child) for child in tree)
        case set():
            return {map_specs(fn, child) for child in tree}
        case frozenset():
            return frozenset(map_specs(fn, child) for child in tree)
        case PydanticBaseModel():
            names = type(tree).model_fields
            return {name: map_specs(fn, getattr(tree, name)) for name in names}
        case _ if is_dataclass(tree) and not isinstance(tree, type):
            names = [field.name for field in fields(tree)]
            return {name: map_specs(fn, getattr(tree, name)) for name in names}
    return tree


def specs_in(tree: object) -> list[Spec]:
    found: list[Spec] = []
    map_specs(found.append, tree)
    return found
