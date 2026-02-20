from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType


class _Missing:
    pass


_MISSING = _Missing()


def _normalize_type_name(type_name: str) -> str:
    if ":" in type_name:
        return type_name.split(":", maxsplit=1)[0]
    return type_name


def _import_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


def _resolve_module_prefixes(type_name: str) -> list[tuple[str, list[str]]]:
    prefixes: list[tuple[str, list[str]]] = []
    parts = type_name.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        if module_name == "":
            continue
        prefixes.append((module_name, parts[index:]))
    return prefixes


@cache
def resolve_type(type_name: str) -> type | None:
    normalized_type_name = _normalize_type_name(type_name)
    if normalized_type_name == "":
        return None

    module_prefixes = _resolve_module_prefixes(normalized_type_name)
    for module_name, attrs in module_prefixes:
        value = _import_module(module_name)
        if value is None:
            continue

        for attr in attrs:
            value = getattr(value, attr, _MISSING)
            if value is _MISSING:
                break
        else:
            if isinstance(value, type):
                return value

    return None
