from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin

from furu._fields import strip_skiphash


def strip_annotated(declared_type: object) -> object:
    declared_type = strip_skiphash(declared_type)
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
