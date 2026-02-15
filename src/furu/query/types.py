from __future__ import annotations

import importlib
import importlib.util
from functools import cache


class _Missing:
    pass


_MISSING = _Missing()


def _normalize_type_name(type_name: str) -> str:
    if ":" in type_name:
        return type_name.split(":", maxsplit=1)[0]
    return type_name


def _resolve_module_prefix(type_name: str) -> tuple[str, list[str]] | None:
    parts = type_name.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        if module_name == "":
            continue
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            continue
        attrs = parts[index:]
        return module_name, attrs
    return None


@cache
def resolve_type(type_name: str) -> type | None:
    normalized_type_name = _normalize_type_name(type_name)
    if normalized_type_name == "":
        return None

    module_prefix = _resolve_module_prefix(normalized_type_name)
    if module_prefix is None:
        return None

    module_name, attrs = module_prefix
    value = importlib.import_module(module_name)
    for attr in attrs:
        value = getattr(value, attr, _MISSING)
        if value is _MISSING:
            return None

    if isinstance(value, type):
        return value
    return None
