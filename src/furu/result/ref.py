from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, cast

from furu.result.codec import Codec, CodecMeta


class _Unloaded:
    pass


_UNLOADED: Final = _Unloaded()


class Ref[T]:
    """A typed, path-backed handle to a heavy value.

    A ``Ref[T]`` has two states. It is *value-backed* when built with
    ``furu.ref(value)`` inside a ``create()`` body — it simply holds the value
    until the bundle is published, at which point it is rebound to point into
    storage. It is *path-backed* when rehydrated from a manifest (or rebound
    after publish): ``load()`` reads the value through the codec on demand and
    memoizes it. The declared field type decides which state a loaded value
    lands in — a field typed ``T`` loads eagerly, a field typed ``Ref[T]``
    yields this handle.
    """

    def __init__(self, value: T) -> None:
        self._value: T | _Unloaded = value
        self._loader: Callable[[], T] | None = None
        self._path: Path | None = None
        self._codec_pin: type[Codec] | None = None
        self._lock = threading.Lock()

    @classmethod
    def _from_value(cls, value: T, *, codec_pin: type[Codec] | None) -> Ref[T]:
        obj = cls(value)
        obj._codec_pin = codec_pin
        return obj

    @classmethod
    def _from_artifact(
        cls,
        *,
        codec: type[Codec],
        artifact_dir: Path,
        metadata: Mapping[str, object],
    ) -> Ref[T]:
        obj = cls.__new__(cls)
        obj._value = _UNLOADED
        obj._loader = lambda: cast(T, codec().load(metadata, artifact_dir))
        obj._path = artifact_dir
        obj._codec_pin = None
        obj._lock = threading.Lock()
        return obj

    def _rebind(
        self,
        *,
        codec: type[Codec],
        artifact_dir: Path,
        metadata: Mapping[str, object],
    ) -> None:
        """Flip a value-backed ref to path-backed, pointing at published storage."""
        with self._lock:
            self._value = _UNLOADED
            self._loader = lambda: cast(T, codec().load(metadata, artifact_dir))
            self._path = artifact_dir
            self._codec_pin = None

    @property
    def is_loaded(self) -> bool:
        return self._value is not _UNLOADED

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("ref path is only available after persistence")
        return self._path

    def __repr__(self) -> str:
        if self._value is _UNLOADED:
            return "Ref(unloaded)"
        return f"Ref({type(self._value).__name__})"

    def load(self) -> T:
        if self._value is not _UNLOADED:
            return cast(T, self._value)

        with self._lock:
            if self._value is _UNLOADED:
                if self._loader is None:
                    raise RuntimeError("ref has no loader")
                self._value = self._loader()
                self._loader = None
            return cast(T, self._value)


def ref[T](value: T, *, codec: type[Codec] | None = None) -> Ref[T]:
    """Wrap a heavy value as a ``Ref[T]`` field, persisted by a codec.

    Resolves a codec eagerly — the explicit ``codec=`` pin, else the registry —
    and raises here, at the call site, when neither matches, rather than failing
    deep in publish. A value with no registered codec must pass ``codec=``.
    """
    if codec is None and CodecMeta.find_codec(value, ()) is None:
        raise TypeError(
            f"furu.ref() received a value of type {type(value).__name__!r} with "
            "no resolvable codec. Either declare the field as a plain eager "
            "value (not Ref[...]), or pass an explicit codec=..."
        )
    return Ref._from_value(value, codec_pin=codec)
