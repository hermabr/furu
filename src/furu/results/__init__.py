from furu.results import api as _api
from furu.results.codecs import (
    DumpContext,
    JsonFileCodec,
    LoadContext,
    NumpyNpyCodec,
    PolarsParquetCodec,
    ResultCodec,
)
from furu.results.errors import (
    ResultDeserializationError,
    ResultPathCollisionError,
    ResultPersistenceError,
    ResultSerializationError,
    UnknownResultCodecError,
)
from furu.results.io import dump_result_bundle, load_result_bundle
from furu.results.lazy import LazyValue
from furu.results.registry import ResultRegistry

FuruResult = _api.FuruResult
ResultConfig = _api.ResultConfig
SaveWith = _api.SaveWith
lazy = _api.lazy
result = _api.result
result_at = _api.result_at
result_when_type = _api.result_when_type

__all__ = [
    "DumpContext",
    "FuruResult",
    "JsonFileCodec",
    "LazyValue",
    "LoadContext",
    "NumpyNpyCodec",
    "PolarsParquetCodec",
    "ResultCodec",
    "ResultConfig",
    "ResultDeserializationError",
    "ResultPathCollisionError",
    "ResultPersistenceError",
    "ResultRegistry",
    "ResultSerializationError",
    "SaveWith",
    "UnknownResultCodecError",
    "dump_result_bundle",
    "lazy",
    "load_result_bundle",
    "result",
    "result_at",
    "result_when_type",
]
