from __future__ import annotations

import chz

from .serialization.serializer import JsonValue


def schema_key_from_furu_obj(furu_obj: dict[str, JsonValue]) -> tuple[str, ...]:
    if not isinstance(furu_obj, dict):
        raise TypeError(f"schema_key requires dict furu_obj, got {type(furu_obj)}")
    keys: set[str] = set()
    for key in furu_obj:
        if not isinstance(key, str):
            raise TypeError(f"schema_key requires string keys, got {type(key)}")
        if key.startswith("_"):
            continue
        keys.add(key)
    return tuple(sorted(keys))


def schema_key_from_metadata_raw(metadata: dict[str, JsonValue]) -> tuple[str, ...]:
    raw = metadata.get("schema_key")
    if raw is None:
        raise ValueError("metadata missing schema_key")
    if isinstance(raw, tuple):
        items = raw
    elif isinstance(raw, list):
        items = raw
    else:
        raise TypeError(f"metadata schema_key must be list or tuple, got {type(raw)}")
    keys: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"metadata schema_key must be strings, got {type(item)}")
        keys.append(item)
    return tuple(keys)


def schema_key_from_cls(cls: type) -> tuple[str, ...]:
    fields = chz.chz_fields(cls)
    keys = {
        field.logical_name
        for field in fields.values()
        if not field.logical_name.startswith("_")
    }
    return tuple(sorted(keys))
