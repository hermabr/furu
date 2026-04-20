from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from furu.results.paths import LogicalPath, format_logical_path
from furu.utils import JsonValue

if TYPE_CHECKING:
    from furu.results.walker import DumpContext, LoadContext


class ResultCodec(Protocol):
    codec_id: str

    def dump(
        self,
        value: object,
        artifact_dir: Path,
        ctx: "DumpContext",
    ) -> dict[str, JsonValue] | None: ...

    def load(
        self,
        artifact_dir: Path,
        meta: dict[str, JsonValue] | None,
        ctx: "LoadContext",
    ) -> object: ...


class ResultRegistry:
    __slots__ = ("_codecs", "_type_defaults")

    def __init__(self) -> None:
        self._codecs: dict[str, ResultCodec] = {}
        self._type_defaults: dict[type[object], str] = {}

    def register_codec(self, codec: ResultCodec) -> None:
        self._codecs[codec.codec_id] = codec

    def get_codec(
        self,
        codec_id: str,
        *,
        logical_path: LogicalPath = (),
    ) -> ResultCodec:
        try:
            return self._codecs[codec_id]
        except KeyError as exc:
            raise ValueError(
                f"missing result codec {codec_id!r} at {format_logical_path(logical_path)}"
            ) from exc

    def register_default_codec(
        self,
        tp: type[object],
        codec: str | ResultCodec,
    ) -> None:
        codec_id = ensure_codec_id(codec, self)
        self._type_defaults[tp] = codec_id

    def resolve_default_codec_id(self, value: object | type[object]) -> str | None:
        tp = value if isinstance(value, type) else type(value)
        if not isinstance(tp, type):
            return None
        for candidate in tp.__mro__:
            codec_id = self._type_defaults.get(candidate)
            if codec_id is not None:
                return codec_id
        return None


def ensure_codec_id(codec: str | ResultCodec, registry: ResultRegistry) -> str:
    if isinstance(codec, str):
        return codec
    registry.register_codec(codec)
    return codec.codec_id


def default_result_registry() -> ResultRegistry:
    from furu.results.codecs.json_file import JsonFileCodec
    from furu.results.codecs.pickle import PickleCodec

    registry = ResultRegistry()
    registry.register_codec(JsonFileCodec())
    registry.register_codec(PickleCodec())

    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        from furu.results.codecs.numpy_npy import NumpyNpyCodec

        numpy_codec = NumpyNpyCodec()
        registry.register_codec(numpy_codec)
        registry.register_default_codec(np.ndarray, numpy_codec)

    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        from furu.results.codecs.polars_parquet import PolarsParquetCodec

        polars_codec = PolarsParquetCodec()
        registry.register_codec(polars_codec)
        registry.register_default_codec(pl.DataFrame, polars_codec)

    return registry
