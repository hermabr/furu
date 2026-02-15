"""Type resolution helpers for query AST type operators."""

from __future__ import annotations

import builtins
import importlib


_TYPE_CACHE: dict[str, type | None] = {}


def resolve_type(type_path: str) -> type | None:
    """Resolve ``type_path`` to a Python class.

    Supports fully-qualified names and enum-like strings of the form
    ``"module.Type:VALUE"`` by stripping the ``":VALUE"`` suffix.

    The resolver repeatedly shortens the module prefix until a module can be
    imported, then resolves any remaining attributes on that module.
    """

    if type_path in _TYPE_CACHE:
        return _TYPE_CACHE[type_path]

    trimmed = type_path.split(":", 1)[0]
    parts = trimmed.split(".")

    if not parts or not parts[0]:
        _TYPE_CACHE[type_path] = None
        return None

    resolved = _resolve_without_cache(parts)
    _TYPE_CACHE[type_path] = resolved
    return resolved


def _resolve_without_cache(parts: list[str]) -> type | None:
    for start in range(len(parts), 0, -1):
        module_name = ".".join(parts[:start])
        attrs = parts[start:]

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        current: object = module
        for attr in attrs:
            if not hasattr(current, attr):
                current = None
                break
            current = getattr(current, attr)

        if isinstance(current, type):
            return current

    if len(parts) == 1:
        if hasattr(builtins, parts[0]):
            current = getattr(builtins, parts[0])
            if isinstance(current, type):
                return current

    return None


__all__ = ["resolve_type"]
