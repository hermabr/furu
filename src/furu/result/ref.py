from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from furu.result.codec import Codec, CodecMeta


class _Unloaded:
    pass


_UNLOADED: Final = _Unloaded()


class Ref[T]:
    def __init__(
        self,
        value: T | _Unloaded,
        *,
        codec: type[Codec[T]],
        artifact_dir: Path | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._value = value
        self._codec = codec
        self._artifact_dir = artifact_dir
        self._metadata = dict(metadata or {})
        self._lock = threading.Lock()

    @classmethod
    def _from_artifact(
        cls,
        *,
        codec: type[Codec[T]],
        artifact_dir: Path,
        metadata: Mapping[str, object],
    ) -> Ref[T]:
        return cls(_UNLOADED, codec=codec, artifact_dir=artifact_dir, metadata=metadata)

    @property
    def is_loaded(self) -> bool:
        return self._value is not _UNLOADED

    @property
    def path(self) -> Path:
        if self._artifact_dir is None:
            raise RuntimeError("ref path is only available after persistence")
        return self._artifact_dir

    def _rebind(self, *, artifact_dir: Path, metadata: Mapping[str, object]) -> None:
        with self._lock:
            self._value = _UNLOADED
            self._artifact_dir = artifact_dir
            self._metadata = dict(metadata)

    def load(self) -> T:
        if self._value is not _UNLOADED:
            return cast(T, self._value)
        with self._lock:
            if self._value is _UNLOADED:
                if self._artifact_dir is None:
                    raise RuntimeError("ref has no persisted artifact")
                self._value = self._codec().load(self._metadata, self._artifact_dir)
            return cast(T, self._value)

    def __repr__(self) -> str:
        return (
            "Ref(unloaded)"
            if self._value is _UNLOADED
            else f"Ref({type(self._value).__name__})"
        )


def ref[T](value: T, *, codec: type[Codec[T]] | None = None) -> Ref[T]:
    resolved = codec or CodecMeta.find_codec(value, ())
    if resolved is None:
        raise ValueError(
            "furu.ref() requires a codec-backed value; use a plain eager field "
            "for inline JSON data or pass an explicit codec=."
        )
    return Ref(value, codec=resolved)
