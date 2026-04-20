from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol, Self

from furu.utils import JsonValue, fully_qualified_name

if TYPE_CHECKING:
    from furu.results.context import DumpContext, LoadContext


class SupportsFuruResult(Protocol):
    def __furu_save_result__(self, ctx: "DumpContext") -> JsonValue: ...

    @classmethod
    def __furu_load_result__(cls, node: JsonValue, ctx: "LoadContext") -> Self: ...


class FuruResult:
    pass


def supports_furu_result_protocol(value: object) -> bool:
    return callable(getattr(value, "__furu_save_result__", None)) and callable(
        getattr(type(value), "__furu_load_result__", None)
    )


def protocol_type_name(tp: type[object]) -> str:
    return fully_qualified_name(tp)


def resolve_type_name(type_name: str) -> type[object]:
    module_name, _, qualname = type_name.rpartition(".")
    if not module_name or not qualname:
        raise TypeError(f"Invalid Python type reference {type_name!r}")

    obj: object = import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)

    if not isinstance(obj, type):
        raise TypeError(f"{type_name!r} did not resolve to a Python type")
    return obj
