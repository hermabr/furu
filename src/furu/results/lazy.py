from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from furu.results.paths import LogicalPath
from furu.utils import JsonValue

T = TypeVar("T")

_UNSET = object()


@dataclass(slots=True)
class LazyValue(Generic[T]):
    _loader: Callable[[], T]
    _path: LogicalPath
    _serializer: str
    _meta: JsonValue | None = None
    _python_type: str | None = None
    _cached: object = field(default=_UNSET, init=False, repr=False)

    def load(self) -> T:
        if self._cached is _UNSET:
            self._cached = self._loader()
        return cast(T, self._cached)

    @property
    def meta(self) -> JsonValue | None:
        return self._meta

    def __repr__(self) -> str:
        loaded = self._cached is not _UNSET
        type_label = self._python_type or "unknown"
        return (
            "LazyValue("
            f"path={self._path.display()}, "
            f"serializer={self._serializer!r}, "
            f"type={type_label!r}, "
            f"loaded={loaded}"
            ")"
        )
