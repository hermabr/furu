from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ResultCodecError
from .protocol import ResultCodec


@dataclass(slots=True)
class ResultRegistry:
    _codecs: dict[str, ResultCodec[Any]] = field(default_factory=dict)
    _types: dict[type[Any], str] = field(default_factory=dict)

    def register_codec(self, codec: ResultCodec[Any]) -> None:
        self._codecs[codec.codec_id] = codec

    def register_type(self, tp: type[Any], codec: str | ResultCodec[Any]) -> None:
        if not isinstance(tp, type):
            raise TypeError("register_type() expects a concrete Python type")
        self._types[tp] = self._normalize_codec_ref(codec)

    def with_type(
        self, tp: type[Any], codec: str | ResultCodec[Any]
    ) -> "ResultRegistry":
        cloned = ResultRegistry(
            _codecs=self._codecs.copy(),
            _types=self._types.copy(),
        )
        cloned.register_type(tp, codec)
        return cloned

    def get_codec(self, codec_id: str) -> ResultCodec[Any]:
        try:
            return self._codecs[codec_id]
        except KeyError as exc:
            raise ResultCodecError(f"unknown result codec {codec_id!r}") from exc

    def resolve_for_value(self, value: Any) -> ResultCodec[Any] | None:
        value_type = type(value)
        exact = self._types.get(value_type)
        if exact is not None:
            return self.get_codec(exact)

        for candidate in value_type.__mro__[1:]:
            codec_id = self._types.get(candidate)
            if codec_id is not None:
                return self.get_codec(codec_id)

        virtual_matches = [
            tp
            for tp in self._types
            if _matches_virtual_base(value, tp) and tp not in value_type.__mro__
        ]
        if not virtual_matches:
            return None

        most_specific = [
            tp
            for tp in virtual_matches
            if not any(
                tp is not other and issubclass(other, tp) for other in virtual_matches
            )
        ]
        if len(most_specific) != 1:
            labels = ", ".join(sorted(tp.__name__ for tp in most_specific))
            raise ResultCodecError(
                f"ambiguous result codec for {value_type.__module__}.{value_type.__qualname__}: {labels}"
            )

        return self.get_codec(self._types[most_specific[0]])

    def _normalize_codec_ref(self, codec: str | ResultCodec[Any]) -> str:
        if isinstance(codec, str):
            return codec
        self.register_codec(codec)
        return codec.codec_id


def _matches_virtual_base(value: Any, tp: type[Any]) -> bool:
    try:
        return isinstance(value, tp)
    except TypeError:
        return False
