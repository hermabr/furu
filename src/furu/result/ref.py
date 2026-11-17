from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Final, cast

from furu.result.codec import Codec, CodecMeta


class _Unloaded:
    pass


_UNLOADED: Final = _Unloaded()


class Ref[T]:
    _value: T | _Unloaded
    _codec: type[Codec[Any]]
    _read: Callable[[], T] | None
    _lock: threading.Lock

    def __init__(self, value: T, *, codec: type[Codec[Any]]) -> None:
        self._value = value
        self._codec = codec
        self._read = None
        self._lock = threading.Lock()

    @classmethod
    def _from_stored(cls, *, codec: type[Codec[Any]], read: Callable[[], T]) -> Ref[T]:
        obj = cls.__new__(cls)
        obj._value = _UNLOADED
        obj._codec = codec
        obj._read = read
        obj._lock = threading.Lock()
        return obj

    def __repr__(self) -> str:
        if self._value is _UNLOADED:
            return "Ref(unloaded)"
        return f"Ref({type(self._value).__name__})"

    def load(self) -> T:
        if self._value is not _UNLOADED:
            return cast(T, self._value)

        with self._lock:
            if self._value is _UNLOADED:
                assert self._read is not None
                self._value = self._read()
            return cast(T, self._value)


def ref[T](value: T, *, codec: type[Codec[T]] | None = None) -> Ref[T]:
    if codec is None:
        codec = CodecMeta.find_codec(value, ())
        if codec is None:
            raise TypeError(
                f"furu.ref() found no codec for value of type "
                f"{type(value).__name__!r}: a Ref is stored as a separate "
                "artifact, which requires a codec. Pass codec=... explicitly, "
                "or drop the Ref and declare the field as a plain eager value."
            )
    return Ref(value, codec=codec)
