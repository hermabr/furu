from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Final, cast


class _Unloaded:
    pass


_UNLOADED: Final = _Unloaded()


class LazyResult[T]:
    def __init__(self, value: T) -> None:
        self._value: T | _Unloaded = value
        self._loader: Callable[[], T] | None = None
        self._path: Path | None = None
        self._lock = threading.Lock()

    @classmethod
    def _from_loader(cls, loader: Callable[[], T], *, path: Path) -> LazyResult[T]:
        obj = cls.__new__(cls)
        obj._value = _UNLOADED
        obj._loader = loader
        obj._path = path
        obj._lock = threading.Lock()
        return obj

    @property
    def is_loaded(self) -> bool:
        return self._value is not _UNLOADED

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("lazy result path is only available after persistence")
        return self._path

    def __repr__(self) -> str:
        if self._value is _UNLOADED:
            return "LazyResult(unloaded)"
        return f"LazyResult({type(self._value).__name__})"

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
