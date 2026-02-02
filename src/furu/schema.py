from __future__ import annotations

import chz

from .serialization.serializer import JsonValue


def schema_key_from_furu_obj(furu_obj: dict[str, JsonValue]) -> tuple[str, ...]:
    """
    Derives a schema key tuple from a furu object by collecting its public property names.
    
    Parameters:
        furu_obj (dict[str, JsonValue]): Mapping representing a furu object.
    
    Returns:
        tuple[str, ...]: Sorted tuple of unique keys from furu_obj that do not start with "_".
    
    Raises:
        TypeError: If furu_obj is not a dict or any key is not a string.
    """
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
    """
    Extracts a schema key tuple from raw metadata.
    
    Parameters:
        metadata (dict[str, JsonValue]): Mapping that must contain a "schema_key" entry whose value is a list or tuple of strings.
    
    Returns:
        tuple[str, ...]: The sequence of schema key strings in the same order as provided in metadata.
    
    Raises:
        ValueError: If "schema_key" is missing from metadata.
        TypeError: If "schema_key" is not a list or tuple, or if any item in it is not a string.
    """
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
    """
    Derives a schema key from a class by extracting its public field logical names.
    
    Parameters:
        cls (type): Class whose declared fields will be inspected.
    
    Returns:
        tuple[str, ...]: Sorted tuple of unique logical names for the class's fields whose `logical_name` does not start with an underscore.
    """
    fields = chz.chz_fields(cls)
    keys = {
        field.logical_name
        for field in fields.values()
        if not field.logical_name.startswith("_")
    }
    return tuple(sorted(keys))