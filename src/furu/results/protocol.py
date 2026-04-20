from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from furu.results.nodes import ManifestNode
    from furu.results.walker import DumpContext, LoadContext


class SupportsFuruResult(Protocol):
    def __furu_result_dump__(self, ctx: "DumpContext") -> "ManifestNode": ...

    @classmethod
    def __furu_result_load__(cls, node: "ManifestNode", ctx: "LoadContext") -> Self: ...


class FuruResult:
    def __furu_result_dump__(self, ctx: "DumpContext") -> "ManifestNode":
        raise NotImplementedError

    @classmethod
    def __furu_result_load__(cls, node: "ManifestNode", ctx: "LoadContext") -> Self:
        raise NotImplementedError
