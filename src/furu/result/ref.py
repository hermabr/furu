from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from furu.result.codec import Codec


class _Unloaded:
    pass


_UNLOADED: Final = _Unloaded()


class Ref[T]:
    """Opt-in lazy handle to a codec-backed artifact.

    Value-backed after ``furu.ref(value)``; path-backed once persisted (rehydrated
    from a manifest, or rebound in place after its bundle is published).
    """

    _value: T | _Unloaded
    _loader: Callable[[], T] | None
    _codec: type[Codec]
    _path: Path | None
    _lock: threading.Lock

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("construct a Ref via furu.ref(value)")

    @classmethod
    def _from_value(cls, value: T, codec: type[Codec]) -> Ref[T]:
        obj = cls.__new__(cls)
        obj._value = value
        obj._loader = None
        obj._codec = codec
        obj._path = None
        obj._lock = threading.Lock()
        return obj

    @classmethod
    def _from_storage(
        cls,
        *,
        artifact_dir: Path,
        codec: type[Codec],
        metadata: Mapping[str, object],
    ) -> Ref[T]:
        obj = cls.__new__(cls)
        obj._value = _UNLOADED
        obj._loader = lambda: cast(T, codec().load(metadata, artifact_dir))
        obj._codec = codec
        obj._path = artifact_dir
        obj._lock = threading.Lock()
        return obj

    def _bind_to_storage(
        self, artifact_dir: Path, codec: type[Codec], metadata: Mapping[str, object]
    ) -> None:
        with self._lock:
            self._value = _UNLOADED
            self._loader = lambda: cast(T, codec().load(metadata, artifact_dir))
            self._codec = codec
            self._path = artifact_dir

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("Ref path is only available after persistence")
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
                    raise RuntimeError("Ref has no loader")
                self._value = self._loader()
                self._loader = None
            return cast(T, self._value)


def ref[T](value: T, *, codec: type[Codec] | None = None) -> Ref[T]:
    from furu.result.codec import CodecMeta

    resolved = codec or CodecMeta.find_codec(value, ())
    if resolved is None:
        raise TypeError(
            f"furu.ref() received a {type(value).__name__} with no resolvable codec. "
            "Declare the field as a plain (eager) field instead, or pass an explicit "
            "furu.ref(value, codec=...)."
        )
    return Ref._from_value(value, resolved)
