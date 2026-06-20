from __future__ import annotations

from typing import Annotated, Final, get_args, get_origin


class _SkipHash:
    __slots__ = ()

    def __repr__(self) -> str:
        return "furu.skip_hash"


skip_hash: Final = _SkipHash()
"""Mark a field so it is kept in the schema and artifact data but excluded from
the schema and artifact hashes (and therefore from the object identity)."""


def has_skip_hash(declared_type: object) -> bool:
    if get_origin(declared_type) is not Annotated:
        return False
    return any(metadata is skip_hash for metadata in get_args(declared_type)[1:])
