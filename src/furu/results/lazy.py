from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from furu.utils import JsonValue

if TYPE_CHECKING:
    from furu.results.codecs import LoadContext, ResultCodec

T = TypeVar("T")


@dataclass(slots=True)
class LazyValue(Generic[T]):
    _codec: "ResultCodec[T]"
    _ctx: "LoadContext"
    _meta: JsonValue | None = None
    _loaded: bool = field(default=False, init=False)
    _value: T | None = field(default=None, init=False)

    def load(self, *, memoize: bool = True) -> T:
        if self._loaded:
            return self._value  # type: ignore[return-value]

        value = self._codec.load(self._ctx, self._meta)
        if memoize:
            self._loaded = True
            self._value = value
        return value

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def meta(self) -> JsonValue | None:
        return self._meta

    @property
    def artifact_dir(self) -> Path:
        return self._ctx.artifact_dir

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "unloaded"
        return (
            "LazyValue("
            f"codec={self._codec.codec_id!r}, "
            f"artifact_dir={str(self._ctx.artifact_dir)!r}, "
            f"state={state!r}, "
            f"meta={self._meta!r})"
        )
