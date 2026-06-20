from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin


class _SkipHashAnnotation:
    __slots__ = ()

    def __repr__(self) -> str:
        return "furu.skip_hash"


skip_hash = _SkipHashAnnotation()


def has_skip_hash(declared_type: object) -> bool:
    if get_origin(declared_type) is not Annotated:
        return False
    return any(metadata is skip_hash for metadata in get_args(declared_type)[1:])


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
