from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar, cast

T = TypeVar("T")

_MISSING = object()


class LazyValue(Generic[T]):
    __slots__ = ("_loader", "_cached_value", "_meta", "_codec")

    def __init__(
        self,
        *,
        loader: Callable[[], T] | None = None,
        value: T | object = _MISSING,
        meta: object | None = None,
        codec: object | None = None,
    ) -> None:
        self._loader = loader
        self._cached_value = value
        self._meta = meta
        self._codec = codec

    @classmethod
    def from_value(cls, value: T, *, codec: object | None = None) -> "LazyValue[T]":
        return cls(value=value, codec=codec)

    @classmethod
    def from_loader(
        cls,
        loader: Callable[[], T],
        *,
        meta: object | None = None,
        codec: object | None = None,
    ) -> "LazyValue[T]":
        return cls(loader=loader, meta=meta, codec=codec)

    def load(self, *, cache: bool = True) -> T:
        if self._cached_value is not _MISSING:
            return cast(T, self._cached_value)
        if self._loader is None:
            raise RuntimeError("lazy value has no loader")
        value = self._loader()
        if cache:
            self._cached_value = value
        return value

    @property
    def is_loaded(self) -> bool:
        return self._cached_value is not _MISSING

    @property
    def meta(self) -> object | None:
        return self._meta

    @property
    def codec(self) -> object | None:
        return self._codec

    def value_for_save(self) -> T:
        return self.load(cache=True)

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "pending"
        return f"LazyValue({status})"
