from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Final, cast


class _Unloaded:
    pass


_UNLOADED: Final = _Unloaded()


class LazyResult[T]:
    def __init__(self, value: T) -> None:
        self._value: T | _Unloaded = value
        self._loader: Callable[[], T] | None = None
        self._lock = threading.Lock()

    @classmethod
    def _from_loader(cls, loader: Callable[[], T]) -> LazyResult[T]:
        obj = cls.__new__(cls)
        obj._value = _UNLOADED
        obj._loader = loader
        obj._lock = threading.Lock()
        return obj

    @property
    def is_loaded(self) -> bool:
        return self._value is not _UNLOADED

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "unloaded"
        return f"{type(self).__name__}({state})"

    def load(self) -> T:
        if self._value is not _UNLOADED:
            return cast(T, self._value)

        with self._lock:
            if self._value is _UNLOADED:
                if self._loader is None:
                    raise RuntimeError("lazy result has no loader")
                self._value = self._loader()
                self._loader = None
            return cast(T, self._value)
