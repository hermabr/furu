from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Annotated, Any, Final, TypeVar, get_args, get_origin

if TYPE_CHECKING:
    from furu.core import Spec


class _SkipHash:
    __slots__ = ()

    def __repr__(self) -> str:
        return "furu.skip_hash"


skip_hash: Final = _SkipHash()


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


@cache
def declared_result_type(spec_type: type[Spec[Any]]) -> object:
    from furu.core import Spec

    declared: object = Any
    for cls in spec_type.__mro__:
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Spec:
                declared = get_args(base)[0]
                break
        else:
            continue
        break

    if isinstance(declared, TypeVar) or any(
        isinstance(arg, TypeVar) for arg in get_args(declared)
    ):
        raise TypeError(
            f"{spec_type.__name__} must declare its concrete result type directly as Spec[...]"
        )
    return declared
