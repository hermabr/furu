from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Protocol, Self

from furu.results.codecs import (
    JsonTreeCodec,
    NumpyArrayCodec,
    PickleCodec,
    PolarsDataFrameCodec,
)
from furu.results.errors import UnknownResultCodecError


class ResultCodec(Protocol):
    codec_id: str

    def dump(self, value: Any, ctx) -> Any: ...

    def load(self, ctx, meta: Any) -> Any: ...


@dataclass(frozen=True)
class ResultRegistry:
    codecs: tuple[ResultCodec, ...] = ()
    type_bindings: tuple[tuple[type[Any], str], ...] = ()

    @classmethod
    def default(cls) -> Self:
        registry = cls().with_codec(JsonTreeCodec()).with_codec(PickleCodec())
        if importlib.util.find_spec("numpy") is not None:
            np = importlib.import_module("numpy")
            registry = registry.with_codec(NumpyArrayCodec(), types=(np.ndarray,))
        if importlib.util.find_spec("polars") is not None:
            pl = importlib.import_module("polars")
            registry = registry.with_codec(
                PolarsDataFrameCodec(),
                types=(pl.DataFrame,),
            )
        return registry

    def with_codec(
        self,
        codec: ResultCodec,
        *,
        types: tuple[type[Any], ...] = (),
    ) -> Self:
        codecs = tuple(
            existing for existing in self.codecs if existing.codec_id != codec.codec_id
        ) + (codec,)
        type_bindings = tuple(
            binding for binding in self.type_bindings if binding[1] != codec.codec_id
        ) + tuple((value_type, codec.codec_id) for value_type in types)
        return type(self)(codecs=codecs, type_bindings=type_bindings)

    def get_codec(self, codec_id: str) -> ResultCodec:
        for codec in reversed(self.codecs):
            if codec.codec_id == codec_id:
                return codec
        raise UnknownResultCodecError(f"Unknown result codec: {codec_id}")

    def find_codec_id_for_value(self, value: Any) -> str | None:
        for value_type, codec_id in reversed(self.type_bindings):
            if isinstance(value, value_type):
                return codec_id
        return None
