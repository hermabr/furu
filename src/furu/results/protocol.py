from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from furu.results.paths import LogicalPath
from furu.utils import JsonValue

if TYPE_CHECKING:
    from furu.results.api import ResultConfig
    from furu.results.registry import ResultRegistry


@dataclass(frozen=True)
class DumpContext:
    bundle_dir: Path
    artifact_dir: Path
    logical_path: LogicalPath
    registry: ResultRegistry
    config: ResultConfig


@dataclass(frozen=True)
class LoadContext:
    bundle_dir: Path
    artifact_dir: Path
    logical_path: LogicalPath
    registry: ResultRegistry
    config: ResultConfig


@runtime_checkable
class SupportsFuruResult(Protocol):
    def __furu_result_dump__(self, ctx: DumpContext) -> JsonValue: ...

    @classmethod
    def __furu_result_load__(cls, ctx: LoadContext, meta: JsonValue) -> Self: ...


class FuruResult:
    def __furu_result_dump__(self, ctx: DumpContext) -> JsonValue:
        raise NotImplementedError

    @classmethod
    def __furu_result_load__(cls, ctx: LoadContext, meta: JsonValue) -> Self:
        raise NotImplementedError


def supports_furu_result_protocol(value: Any) -> bool:
    dump = getattr(value, "__furu_result_dump__", None)
    load = getattr(type(value), "__furu_result_load__", None)
    return callable(dump) and callable(load)
