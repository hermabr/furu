from __future__ import annotations

from collections.abc import Mapping

JsonScalar = str | int | float | bool | None
JsonValue = (
    JsonScalar | dict[str, "JsonValue"] | list["JsonValue"] | tuple["JsonValue", ...]
)


class _PathMissingType:
    def __repr__(self) -> str:
        return "PATH_MISSING"


PATH_MISSING = _PathMissingType()


def get_path(doc: dict[str, JsonValue], path: str) -> JsonValue | _PathMissingType:
    if path == "":
        return doc

    current: JsonValue | dict[str, JsonValue] = doc
    for segment in path.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                return PATH_MISSING
            current = current[segment]
            continue

        if isinstance(current, (list, tuple)) and segment.isdigit():
            index = int(segment)
            if index >= len(current):
                return PATH_MISSING
            current = current[index]
            continue

        return PATH_MISSING

    return current
