from __future__ import annotations

import json
import math
import os
from importlib import import_module
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast, Protocol, TypeVar

from furu.results.errors import ResultDeserializationError, ResultSerializationError
from furu.results.paths import LogicalPath
from furu.utils import JsonValue, import_fully_qualified_name

if TYPE_CHECKING:
    from furu.results.api import ResultConfig
    from furu.results.registry import ResultRegistry

T = TypeVar("T")


def _write_strict_json(path: Path, value: JsonValue) -> None:
    text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.rename(path)


def _read_strict_json(path: Path) -> JsonValue:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_json_compatible(value: object, logical_path: LogicalPath) -> JsonValue:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResultSerializationError(
                "Non-finite floats are not valid JSON values",
                logical_path=logical_path,
            )
        return value
    if isinstance(value, list):
        return [
            _ensure_json_compatible(item, logical_path.child_index(index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        converted: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResultSerializationError(
                    f"JSON-file codec requires string dict keys; got {key!r}",
                    logical_path=logical_path,
                )
            converted[key] = _ensure_json_compatible(item, logical_path.child_key(key))
        return converted
    raise ResultSerializationError(
        "JSON-file codec only supports strict JSON-compatible values",
        logical_path=logical_path,
    )


class ResultCodec(Protocol[T]):
    codec_id: str

    def dump(self, value: T, ctx: "DumpContext") -> JsonValue | None:
        """Write files under ctx.artifact_dir and return JSON metadata."""

    def load(self, ctx: "LoadContext", meta: JsonValue | None) -> T:
        """Load a value from ctx.artifact_dir."""


@dataclass(frozen=True, slots=True)
class DumpContext:
    bundle_dir: Path
    artifact_dir: Path
    logical_path: LogicalPath
    registry: "ResultRegistry"
    config: "ResultConfig"

    def child_field(self, name: str) -> "DumpContext":
        logical_path = self.logical_path.child_field(name)
        return replace(
            self,
            logical_path=logical_path,
            artifact_dir=self.bundle_dir / logical_path.artifact_relative_dir(),
        )

    def child_key(self, key: str) -> "DumpContext":
        logical_path = self.logical_path.child_key(key)
        return replace(
            self,
            logical_path=logical_path,
            artifact_dir=self.bundle_dir / logical_path.artifact_relative_dir(),
        )

    def child_index(self, index: int) -> "DumpContext":
        logical_path = self.logical_path.child_index(index)
        return replace(
            self,
            logical_path=logical_path,
            artifact_dir=self.bundle_dir / logical_path.artifact_relative_dir(),
        )


@dataclass(frozen=True, slots=True)
class LoadContext:
    bundle_dir: Path
    artifact_dir: Path
    logical_path: LogicalPath
    registry: "ResultRegistry"
    config: "ResultConfig"
    node: dict[str, JsonValue] | None = None

    def child_field(self, name: str) -> "LoadContext":
        logical_path = self.logical_path.child_field(name)
        return replace(self, logical_path=logical_path)

    def child_key(self, key: str) -> "LoadContext":
        logical_path = self.logical_path.child_key(key)
        return replace(self, logical_path=logical_path)

    def child_index(self, index: int) -> "LoadContext":
        logical_path = self.logical_path.child_index(index)
        return replace(self, logical_path=logical_path)

    def for_external(
        self,
        *,
        artifact_dir: Path,
        node: dict[str, JsonValue],
    ) -> "LoadContext":
        return replace(self, artifact_dir=artifact_dir, node=node)


class JsonFileCodec:
    codec_id = "furu.json.v1"

    def dump(self, value: object, ctx: DumpContext) -> JsonValue | None:
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        json_value = _ensure_json_compatible(value, ctx.logical_path)
        _write_strict_json(ctx.artifact_dir / "value.json", json_value)
        return {"storage": "json"}

    def load(self, ctx: LoadContext, meta: JsonValue | None) -> JsonValue:
        del meta
        return _read_strict_json(ctx.artifact_dir / "value.json")


class NumpyNpyCodec:
    codec_id = "numpy.ndarray.npy.v1"

    def dump(self, value: object, ctx: DumpContext) -> JsonValue | None:
        np = import_module("numpy")

        if not isinstance(value, np.ndarray):
            raise ResultSerializationError(
                "NumPy codec expected numpy.ndarray",
                logical_path=ctx.logical_path,
            )
        if value.dtype.hasobject:
            raise ResultSerializationError(
                "Object-dtype NumPy arrays are not supported; register a custom codec instead",
                logical_path=ctx.logical_path,
            )

        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(ctx.artifact_dir / "value.npy", value, allow_pickle=False)
        return {"dtype": str(value.dtype), "shape": list(value.shape)}

    def load(self, ctx: LoadContext, meta: JsonValue | None) -> object:
        del meta
        np = import_module("numpy")

        return np.load(ctx.artifact_dir / "value.npy", allow_pickle=False)


class PolarsParquetCodec:
    codec_id = "polars.DataFrame.parquet.v1"

    def dump(self, value: object, ctx: DumpContext) -> JsonValue | None:
        pl = import_module("polars")

        if not isinstance(value, pl.DataFrame):
            raise ResultSerializationError(
                "Polars codec expected polars.DataFrame",
                logical_path=ctx.logical_path,
            )

        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        value.write_parquet(ctx.artifact_dir / "value.parquet")
        return {"columns": value.columns, "height": value.height}

    def load(self, ctx: LoadContext, meta: JsonValue | None) -> object:
        del meta
        pl = import_module("polars")

        return pl.read_parquet(ctx.artifact_dir / "value.parquet")


class ObjectProtocolCodec:
    codec_id = "furu.object-protocol.v1"

    def dump(self, value: object, ctx: DumpContext) -> JsonValue | None:
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        dump = getattr(value, "__furu_result_dump__", None)
        if not callable(dump):
            raise ResultSerializationError(
                "Object protocol value is missing __furu_result_dump__",
                logical_path=ctx.logical_path,
            )
        return cast(JsonValue | None, dump(ctx))

    def load(self, ctx: LoadContext, meta: JsonValue | None) -> object:
        if ctx.node is None:
            raise ResultDeserializationError(
                "Missing protocol manifest payload",
                logical_path=ctx.logical_path,
            )
        python_type = ctx.node.get("python_type")
        if not isinstance(python_type, str):
            raise ResultDeserializationError(
                "Protocol result nodes require python_type",
                logical_path=ctx.logical_path,
            )
        cls = import_fully_qualified_name(python_type)
        load = getattr(cls, "__furu_result_load__", None)
        if not callable(load):
            raise ResultDeserializationError(
                "Object protocol type is missing __furu_result_load__",
                logical_path=ctx.logical_path,
            )
        return load(ctx, meta)


def register_builtin_codecs(registry: "ResultRegistry") -> "ResultRegistry":
    json_codec = JsonFileCodec()
    registry.register_codec(json_codec)

    try:
        np = import_module("numpy")
    except ImportError:
        pass
    else:
        numpy_codec = NumpyNpyCodec()
        registry.register_codec(numpy_codec)
        registry.register_type(np.ndarray, numpy_codec)

    try:
        pl = import_module("polars")
    except ImportError:
        pass
    else:
        polars_codec = PolarsParquetCodec()
        registry.register_codec(polars_codec)
        registry.register_type(pl.DataFrame, polars_codec)

    registry.register_codec(ObjectProtocolCodec())
    return registry
