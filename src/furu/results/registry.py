from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from furu.results.errors import ResultSerializationError, UnknownResultCodecError
from furu.results.paths import LogicalPath

if TYPE_CHECKING:
    from furu.results.codecs import ResultCodec


@dataclass(slots=True)
class ResultRegistry:
    _codecs: dict[str, ResultCodec[Any]] = field(default_factory=dict)
    _type_codecs: dict[type[object], str] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "ResultRegistry":
        registry = cls()
        from furu.results.codecs import register_builtin_codecs

        register_builtin_codecs(registry)
        return registry

    def register_codec(self, codec: "ResultCodec[Any]") -> "ResultRegistry":
        self._codecs[codec.codec_id] = codec
        return self

    def register_type(
        self,
        tp: type[object],
        codec: str | "ResultCodec[Any]",
    ) -> "ResultRegistry":
        codec_id = self._normalize_codec_reference(codec)
        self._type_codecs[tp] = codec_id
        return self

    def _normalize_codec_reference(
        self,
        codec: str | "ResultCodec[Any]",
    ) -> str:
        if isinstance(codec, str):
            if codec not in self._codecs:
                raise UnknownResultCodecError(codec)
            return codec
        self.register_codec(codec)
        return codec.codec_id

    def get_codec(
        self,
        codec_id: str,
        *,
        logical_path: LogicalPath | None = None,
    ) -> "ResultCodec[Any]":
        codec = self._codecs.get(codec_id)
        if codec is None:
            raise UnknownResultCodecError(codec_id, logical_path)
        return codec

    def resolve_type(self, value: object) -> "ResultCodec[Any] | None":
        value_type = type(value)
        matches: list[tuple[int, type[object], str]] = []
        for registered_type, codec_id in self._type_codecs.items():
            try:
                index = value_type.__mro__.index(registered_type)
            except ValueError:
                continue
            matches.append((index, registered_type, codec_id))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0])
        best_index = matches[0][0]
        best = [match for match in matches if match[0] == best_index]
        if len(best) > 1:
            raise ResultSerializationError(
                (
                    f"Ambiguous result codec resolution for {value_type.__module__}."
                    f"{value_type.__qualname__}; register a more specific codec"
                )
            )
        return self.get_codec(best[0][2])
