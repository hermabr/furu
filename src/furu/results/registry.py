from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from furu.utils import JsonValue

from furu.results.codecs.json_file import JsonFileCodec
from furu.results.codecs.numpy_npy import NumpyNpyCodec
from furu.results.codecs.pickle_codec import PickleCodec
from furu.results.codecs.polars_parquet import PolarsParquetCodec
from furu.results.rules import SaveSpec

if TYPE_CHECKING:
    from furu.results.context import DumpContext, LoadContext


class ResultCodec[T](Protocol):
    codec_id: str

    def dump(self, value: T, ctx: "DumpContext", spec: SaveSpec) -> JsonValue: ...

    def load(self, node: JsonValue, ctx: "LoadContext") -> T: ...


@dataclass(frozen=True, slots=True)
class ResultRegistry:
    _codecs: dict[str, ResultCodec[Any]] = field(default_factory=dict)
    _type_defaults: dict[type[object], str] = field(default_factory=dict)

    def get(self, serializer: str | ResultCodec[Any] | None) -> ResultCodec[Any] | None:
        if serializer is None:
            return None
        if isinstance(serializer, str):
            return self._codecs.get(serializer)
        return serializer

    def require(self, serializer: str | ResultCodec[Any]) -> ResultCodec[Any]:
        codec = self.get(serializer)
        if codec is None:
            raise TypeError(f"Unknown result serializer {serializer!r}")
        return codec

    def codec_for_value(self, value: object) -> ResultCodec[Any] | None:
        for tp in type(value).__mro__:
            codec_id = self._type_defaults.get(tp)
            if codec_id is not None:
                return self._codecs[codec_id]
        return None

    def with_codec(
        self,
        codec: ResultCodec[Any],
        *types: type[object],
    ) -> "ResultRegistry":
        codecs = dict(self._codecs)
        type_defaults = dict(self._type_defaults)
        codecs[codec.codec_id] = codec
        for tp in types:
            type_defaults[tp] = codec.codec_id
        return ResultRegistry(_codecs=codecs, _type_defaults=type_defaults)

    @classmethod
    def default(cls) -> "ResultRegistry":
        registry = cls()
        registry = registry.with_codec(cast(ResultCodec[Any], JsonFileCodec()))
        registry = registry.with_codec(cast(ResultCodec[Any], PickleCodec()))

        try:
            import numpy as np
        except ImportError:
            pass
        else:
            registry = registry.with_codec(
                cast(ResultCodec[Any], NumpyNpyCodec()), np.ndarray
            )

        try:
            import polars as pl
        except ImportError:
            pass
        else:
            registry = registry.with_codec(
                cast(ResultCodec[Any], PolarsParquetCodec()), pl.DataFrame
            )

        return registry
