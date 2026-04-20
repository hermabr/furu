from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from furu.utils import JsonValue

from .paths import LogicalPath

if TYPE_CHECKING:
    from .registry import ResultRegistry

T = TypeVar("T")


class ResultCodec(Protocol[T]):
    codec_id: str

    def dump(self, value: T, ctx: "DumpContext") -> JsonValue | None:
        """Write files under ctx.artifact_dir and return optional JSON metadata."""

    def load(self, ctx: "LoadContext") -> T:
        """Read files from ctx.artifact_dir and return the materialized value."""


@dataclass(frozen=True, slots=True)
class DumpContext:
    bundle_dir: Path
    artifact_dir: Path
    artifact_relpath: str
    logical_path: LogicalPath
    registry: "ResultRegistry"


@dataclass(frozen=True, slots=True)
class LoadContext:
    bundle_dir: Path
    artifact_dir: Path
    node: dict[str, JsonValue]
    logical_path: LogicalPath
    registry: "ResultRegistry"


class SupportsFuruResult(Protocol):
    def __furu_result_dump__(self, ctx: DumpContext) -> JsonValue: ...

    @classmethod
    def __furu_result_load__(cls, payload: JsonValue, ctx: LoadContext) -> Any: ...
