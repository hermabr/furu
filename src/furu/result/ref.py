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
        value: T,
        *,
        codec: type[Codec],
        codec_was_explicit: bool,
    ) -> None:
        self._value: T | _Unloaded = value
        self._codec = codec
        self._codec_was_explicit = codec_was_explicit
        self._metadata: Mapping[str, object] | None = None
        self._artifact_dir: Path | None = None
        self._lock = threading.Lock()

    @classmethod
    def _from_artifact(
        cls,
        *,
        codec: type[Codec],
        metadata: Mapping[str, object],
        artifact_dir: Path,
    ) -> Ref[T]:
        obj = cls.__new__(cls)
        obj._value = _UNLOADED
        obj._codec = codec
        obj._codec_was_explicit = True
        obj._metadata = metadata
        obj._artifact_dir = artifact_dir
        obj._lock = threading.Lock()
        return obj

    @property
    def is_loaded(self) -> bool:
        return self._value is not _UNLOADED

    @property
    def path(self) -> Path:
        if self._artifact_dir is None:
            raise RuntimeError("ref path is only available after persistence")
        return self._artifact_dir

    def __repr__(self) -> str:
        if self._value is _UNLOADED:
            return "Ref(unloaded)"
        return f"Ref({type(self._value).__name__})"

    def load(self) -> T:
        if self._value is not _UNLOADED:
            return cast(T, self._value)

        with self._lock:
            if self._value is _UNLOADED:
                if self._metadata is None or self._artifact_dir is None:
                    raise RuntimeError("ref has no persisted artifact")
                self._value = self._codec().load(self._metadata, self._artifact_dir)
            return cast(T, self._value)

    def _value_for_save(self) -> T:
        return self.load()

    def _bind_to_artifact(
        self,
        *,
        codec: type[Codec],
        metadata: Mapping[str, object],
        artifact_dir: Path,
    ) -> None:
        with self._lock:
            self._value = _UNLOADED
            self._codec = codec
            self._codec_was_explicit = True
            self._metadata = metadata
            self._artifact_dir = artifact_dir


def ref[T](value: T, *, codec: type[Codec] | None = None) -> Ref[T]:
    if codec is not None:
        return Ref(value, codec=codec, codec_was_explicit=True)

    inferred_codec = CodecMeta.find_codec(value, ())
    if inferred_codec is None:
        raise TypeError(
            "furu.ref() requires a codec-backed value. Use a plain eager field for "
            "inline manifest values, or pass an explicit codec= for a Ref."
        )
    return Ref(value, codec=inferred_codec, codec_was_explicit=False)
