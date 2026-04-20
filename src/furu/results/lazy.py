from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from furu.utils import JsonValue

if TYPE_CHECKING:
    from furu.results.registry import ResultCodec


class LazyValue[T]:
    __slots__ = ("_cached_value", "_codec", "_is_loaded", "_loader", "_meta")
    _MISSING = object()

    def __init__(
        self,
        *,
        value: T | object = _MISSING,
        loader: Callable[[], T] | None = None,
        meta: JsonValue | None = None,
        codec: str | ResultCodec | None = None,
        is_loaded: bool,
    ) -> None:
        if loader is None and not is_loaded:
            raise ValueError("Unloaded LazyValue requires a loader")
        if is_loaded and value is self._MISSING:
            raise ValueError("Loaded LazyValue requires a value")
        self._cached_value = value
        self._codec = codec
        self._is_loaded = is_loaded
        self._loader = loader
        self._meta = meta

    @classmethod
    def from_value(
        cls,
        value: T,
        *,
        codec: str | ResultCodec | None = None,
    ) -> LazyValue[T]:
        return cls(value=value, codec=codec, is_loaded=True)

    @classmethod
    def from_loader(
        cls,
        loader: Callable[[], T],
        *,
        meta: JsonValue | None = None,
    ) -> LazyValue[T]:
        return cls(loader=loader, meta=meta, is_loaded=False)

    def load(self, *, cache: bool = True) -> T:
        if not cache and self._loader is not None:
            return self._loader()
        if self._is_loaded:
            return cast(T, self._cached_value)
        assert self._loader is not None
        value = self._loader()
        if cache:
            self._cached_value = value
            self._is_loaded = True
        return value

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def meta(self) -> JsonValue | None:
        return self._meta

    @property
    def _furu_wrapped_value(self) -> T | None:
        if self._cached_value is self._MISSING:
            return None
        return cast(T, self._cached_value)

    @property
    def _furu_requested_codec(self) -> str | ResultCodec | None:
        return self._codec

    def _furu_unwrap_for_dump(self) -> T:
        if self._is_loaded:
            return cast(T, self._cached_value)
        return self.load(cache=True)
