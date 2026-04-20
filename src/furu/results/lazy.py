from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, cast

from furu.utils import JsonValue

from .paths import LogicalPath

T = TypeVar("T")


@dataclass(slots=True)
class FuruLazy(Generic[T]):
    _loader: Callable[[], T]
    _logical_path: LogicalPath
    _descriptor: str
    _meta: JsonValue | None = None
    _is_loaded: bool = False
    _value: Any = field(default=None, repr=False)

    @property
    def meta(self) -> JsonValue | None:
        return self._meta

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> T:
        if not self._is_loaded:
            self._value = self._loader()
            self._is_loaded = True
        return cast(T, self._value)

    def __repr__(self) -> str:
        meta = f", meta={self._meta!r}" if self._meta is not None else ""
        return (
            f"FuruLazy({self._descriptor}, path={self._logical_path.display()!r}, "
            f"loaded={self._is_loaded}{meta})"
        )
