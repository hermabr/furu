from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cache
from itertools import chain, repeat
from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
    get_type_hints,
)


@cache
def cached_type_hints(cls: type) -> Mapping[str, Any]:
    return get_type_hints(cls, include_extras=True)


def strip_annotated(declared_type: object) -> object:
    if get_origin(declared_type) is Annotated:
        return get_args(declared_type)[0]
    return declared_type


def child_declared_type(declared_type: object, key: object) -> object:
    declared_type = strip_annotated(declared_type)
    origin = get_origin(declared_type)
    args = get_args(declared_type)
    if origin is list and args:
        return args[0]
    if origin in (set, frozenset) and args:
        return args[0]
    if origin is dict and len(args) == 2:
        return args[1]
    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        if isinstance(key, int) and key < len(args):
            return args[key]
    return Any


def item_declared_types(declared_type: object) -> Iterator[object]:
    """The declared type of each item of a list or tuple, in order.

    Worked out once per container, not per item: one tuple can hold thousands
    of specs.
    """
    declared_type = strip_annotated(declared_type)
    args = get_args(declared_type)
    if get_origin(declared_type) is tuple and args and args[1:] != (Ellipsis,):
        return chain(args, repeat(Any))
    return repeat(child_declared_type(declared_type, 0))
