from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "FuruResult",
    "LazyValue",
    "ResultCodec",
    "ResultRegistry",
    "SaveSpec",
    "SaveWith",
    "at",
    "lazy",
    "load_result_bundle",
    "save_result_bundle",
    "save_with",
    "when_type",
]

if TYPE_CHECKING:
    from furu.results.api import FuruResult, SaveWith, at, lazy, save_with, when_type
    from furu.results.io import load_result_bundle, save_result_bundle
    from furu.results.lazy import LazyValue
    from furu.results.registry import ResultCodec, ResultRegistry
    from furu.results.rules import SaveSpec


def __getattr__(name: str) -> object:
    if name in {"FuruResult", "SaveWith", "at", "lazy", "save_with", "when_type"}:
        from furu.results import api

        return getattr(api, name)
    if name in {"load_result_bundle", "save_result_bundle"}:
        from furu.results import io

        return getattr(io, name)
    if name == "LazyValue":
        from furu.results.lazy import LazyValue

        return LazyValue
    if name in {"ResultCodec", "ResultRegistry"}:
        from furu.results.registry import ResultCodec, ResultRegistry

        return {"ResultCodec": ResultCodec, "ResultRegistry": ResultRegistry}[name]
    if name == "SaveSpec":
        from furu.results.rules import SaveSpec

        return SaveSpec
    raise AttributeError(name)
