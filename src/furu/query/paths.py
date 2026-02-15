"""Path traversal helpers for query document extraction."""

from __future__ import annotations

from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = (
    JSONScalar | list["JSONValue"] | tuple["JSONValue", ...] | dict[str, "JSONValue"]
)


class _PathMissing:
    """Sentinel for missing document paths."""


PATH_MISSING: _PathMissing = _PathMissing()


def get_path(document: dict[str, JSONValue], path: str) -> JSONValue | _PathMissing:
    """Resolve a dot-separated path through nested dict/list structures.

    Returns ``PATH_MISSING`` if any path segment is unavailable.
    """

    current: JSONValue = document
    if not path:
        return document

    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return PATH_MISSING
            current = current[segment]
            continue

        if isinstance(current, (list, tuple)):
            if not segment.isdigit():
                return PATH_MISSING
            index = int(segment)
            if index >= len(current):
                return PATH_MISSING
            current = current[index]
            continue

        return PATH_MISSING

    return current


__all__ = ["PATH_MISSING", "JSONScalar", "JSONValue", "get_path"]
