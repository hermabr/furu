from __future__ import annotations

from importlib import import_module
import json
import math
from typing import Any

from .errors import ResultCodecError
from .protocol import DumpContext, LoadContext
from .registry import ResultRegistry


class JsonFileCodec:
    codec_id = "furu.json.v1"

    def dump(self, value: Any, ctx: DumpContext) -> None:
        _validate_json_compatible(value)
        path = ctx.artifact_dir / "data.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        return None

    def load(self, ctx: LoadContext) -> Any:
        path = ctx.artifact_dir / "data.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


class NumpyArrayCodec:
    codec_id = "numpy.ndarray.npy.v1"

    def dump(self, value: Any, ctx: DumpContext) -> dict[str, object]:
        np = _import_numpy()

        if not isinstance(value, np.ndarray):
            raise ResultCodecError("numpy codec expected a numpy.ndarray")
        if value.dtype == np.dtype("O"):
            raise ResultCodecError("object-dtype numpy arrays are not supported")
        path = ctx.artifact_dir / "data.npy"
        np.save(path, value, allow_pickle=False)
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    def load(self, ctx: LoadContext) -> Any:
        np = _import_numpy()

        return np.load(ctx.artifact_dir / "data.npy", allow_pickle=False)


class PolarsDataFrameCodec:
    codec_id = "polars.dataframe.parquet.v1"

    def dump(self, value: Any, ctx: DumpContext) -> dict[str, object]:
        pl = _import_polars()

        if not isinstance(value, pl.DataFrame):
            raise ResultCodecError("polars codec expected a polars.DataFrame")
        value.write_parquet(ctx.artifact_dir / "data.parquet")
        return {
            "height": value.height,
            "columns": list(value.columns),
        }

    def load(self, ctx: LoadContext) -> Any:
        pl = _import_polars()

        return pl.read_parquet(ctx.artifact_dir / "data.parquet")


def default_result_registry() -> ResultRegistry:
    registry = ResultRegistry()
    registry.register_codec(JsonFileCodec())

    try:
        np = _import_numpy()
    except ImportError:
        pass
    else:
        numpy_codec = NumpyArrayCodec()
        registry.register_codec(numpy_codec)
        registry.register_type(np.ndarray, numpy_codec)

    try:
        pl = _import_polars()
    except ImportError:
        pass
    else:
        polars_codec = PolarsDataFrameCodec()
        registry.register_codec(polars_codec)
        registry.register_type(pl.DataFrame, polars_codec)

    return registry


def _validate_json_compatible(value: Any) -> None:
    match value:
        case None | bool() | int() | str():
            return
        case float():
            if not math.isfinite(value):
                raise ResultCodecError("non-finite floats are not supported in JSON")
            return
        case list():
            for item in value:
                _validate_json_compatible(item)
            return
        case dict():
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ResultCodecError("JSON values require string object keys")
                _validate_json_compatible(item)
            return
        case _:
            raise ResultCodecError(
                f"JSON codec cannot serialize value of type {type(value).__name__}"
            )


def _import_numpy() -> Any:
    return import_module("numpy")


def _import_polars() -> Any:
    return import_module("polars")
