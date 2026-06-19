from __future__ import annotations

from typing import Annotated, Any, Final, get_args, get_origin


class _Skip:
    __slots__ = ()

    def __repr__(self) -> str:
        return "furu.skip"


skip: Final = _Skip()
"""Mark a field as excluded from the artifact and schema via ``Annotated[T, furu.skip]``.

Skipped fields stay normal constructor parameters but never reach the artifact or
schema, so they do not affect an object's hash, ``object_id``, or cached result.
Because a skipped field is absent from the artifact, it cannot be reconstructed when
loading from one, so it must declare a default.
"""


def strip_annotated(declared_type: object) -> object:
    if get_origin(declared_type) is Annotated:
        return get_args(declared_type)[0]
    return declared_type


def is_skipped(declared_type: object) -> bool:
    if get_origin(declared_type) is Annotated:
        return any(metadata is skip for metadata in get_args(declared_type)[1:])
    return False


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
