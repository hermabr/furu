from __future__ import annotations

import dataclasses
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel as PydanticBaseModel

from furu.utils import JsonValue, fully_qualified_name

if TYPE_CHECKING:
    pass


_FURU_KEY = "$furu"
_MANIFEST_NAME = "manifest.json"
_ARTIFACTS_DIRNAME = "artifacts"
_ROOT_ARTIFACT_SEGMENT = "root"


def _is_safe_path_segment(segment: str) -> bool:
    if not isinstance(segment, str):
        return False
    if segment == "":
        return False
    if segment in (".", ".."):
        return False
    if "/" in segment:
        return False
    if "\\" in segment:
        return False
    if "\x00" in segment:
        return False
    return True


@dataclass(frozen=True)
class LogicalPath:
    parts: tuple[object, ...] = ()

    def key(self, key: str) -> "LogicalPath":
        return LogicalPath(self.parts + (("key", key),))

    def index(self, index: int, *, width: int) -> "LogicalPath":
        return LogicalPath(self.parts + (("index", index, width),))

    def field(self, name: str) -> "LogicalPath":
        return LogicalPath(self.parts + (("field", name),))

    def display(self) -> str:
        out = ["$"]
        for part in self.parts:
            kind = part[0]
            if kind == "key":
                key = part[1]
                if isinstance(key, str) and _is_safe_path_segment(key) and "." not in key and "[" not in key and "]" not in key and '"' not in key:
                    out.append(f".{key}")
                else:
                    encoded = json.dumps(key)
                    out.append(f"[{encoded}]")
            elif kind == "index":
                index = part[1]
                width = part[2]
                out.append(f"[{index:0{width}d}]")
            elif kind == "field":
                name = part[1]
                out.append(f".{name}")
            else:
                raise AssertionError(f"unknown logical path part kind: {kind!r}")
        return "".join(out)

    def artifact_dir(self) -> Path:
        if not self.parts:
            return Path(_ARTIFACTS_DIRNAME) / _ROOT_ARTIFACT_SEGMENT
        segments: list[str] = [_ARTIFACTS_DIRNAME]
        for part in self.parts:
            kind = part[0]
            if kind == "key":
                key = part[1]
                if not isinstance(key, str) or not _is_safe_path_segment(key):
                    raise ValueError(
                        f"Unsupported result path at {self.display()}:\n"
                        "dict key cannot be used as an artifact path segment."
                    )
                segments.append(key)
            elif kind == "index":
                index = part[1]
                width = part[2]
                segments.append(f"{index:0{width}d}")
            elif kind == "field":
                name = part[1]
                if not isinstance(name, str) or not _is_safe_path_segment(name):
                    raise ValueError(
                        f"Unsupported result path at {self.display()}:\n"
                        "field name cannot be used as an artifact path segment."
                    )
                segments.append(name)
            else:
                raise AssertionError(f"unknown logical path part kind: {kind!r}")
        return Path(*segments)


@runtime_checkable
class ResultCodec(Protocol):
    @property
    def codec_id(self) -> str: ...

    def matches(self, value: object) -> bool: ...

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue: ...

    def load(
        self,
        *,
        artifact_dir: Path,
        meta: JsonValue,
    ) -> object: ...


class _NumpyNpyCodec:
    codec_id: str = "numpy.ndarray.npy"

    def __init__(self) -> None:
        # Importing here so missing NumPy raises an ImportError that the
        # registry constructor can swallow.
        import numpy  # noqa: F401

    def matches(self, value: object) -> bool:
        try:
            import numpy as np
        except ImportError:
            return False
        return isinstance(value, np.ndarray)

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        import numpy as np

        assert isinstance(value, np.ndarray)
        if value.dtype.hasobject:
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                "numpy object-dtype arrays are not supported by the default npy codec."
            )

        artifact_dir.mkdir(parents=True, exist_ok=False)
        np.save(artifact_dir / "data.npy", value, allow_pickle=False)

        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    def load(
        self,
        *,
        artifact_dir: Path,
        meta: JsonValue,
    ) -> object:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


class _PolarsParquetCodec:
    codec_id: str = "polars.dataframe.parquet"

    def __init__(self) -> None:
        import polars  # noqa: F401

    def matches(self, value: object) -> bool:
        try:
            import polars as pl
        except ImportError:
            return False
        return isinstance(value, pl.DataFrame)

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        import polars as pl

        assert isinstance(value, pl.DataFrame)

        artifact_dir.mkdir(parents=True, exist_ok=False)
        value.write_parquet(artifact_dir / "data.parquet")

        return {
            "rows": value.height,
            "columns": list(value.columns),
        }

    def load(
        self,
        *,
        artifact_dir: Path,
        meta: JsonValue,
    ) -> object:
        import polars as pl

        return pl.read_parquet(artifact_dir / "data.parquet")


class ResultRegistry:
    def __init__(self) -> None:
        self._codecs: list[ResultCodec] = []
        self._by_id: dict[str, ResultCodec] = {}

    def register(self, codec: ResultCodec) -> None:
        if codec.codec_id in self._by_id:
            raise ValueError(f"codec already registered: {codec.codec_id!r}")
        self._codecs.append(codec)
        self._by_id[codec.codec_id] = codec

    def codec_for_value(self, value: object) -> ResultCodec | None:
        for codec in self._codecs:
            if codec.matches(value):
                return codec
        return None

    def codec_by_id(self, codec_id: str) -> ResultCodec:
        if codec_id not in self._by_id:
            raise ValueError(f"unknown result codec: {codec_id!r}")
        return self._by_id[codec_id]


def default_result_registry() -> ResultRegistry:
    registry = ResultRegistry()

    try:
        registry.register(_NumpyNpyCodec())
    except ImportError:
        pass

    try:
        registry.register(_PolarsParquetCodec())
    except ImportError:
        pass

    return registry


def _is_furu_wrapper(value: object) -> bool:
    return (
        isinstance(value, dict)
        and len(value) == 1
        and _FURU_KEY in value
    )


def _is_json_scalar(value: object) -> bool:
    # bool must be checked first since bool is a subclass of int
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, str)):
        return True
    return False


def _resolve_type_id(type_id: str) -> type:
    if "." not in type_id:
        raise ValueError(f"invalid type id: {type_id!r}")
    module_name, _, qualname = type_id.rpartition(".")
    module = importlib.import_module(module_name)
    if not hasattr(module, qualname):
        raise ValueError(
            f"could not resolve type {type_id!r}: "
            f"{module_name} has no attribute {qualname!r}"
        )
    resolved = getattr(module, qualname)
    if not isinstance(resolved, type):
        raise ValueError(f"resolved {type_id!r} is not a type")
    return resolved


def _dump_value(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    # 1. JSON scalar
    if _is_json_scalar(value):
        return value  # type: ignore[return-value]

    # 2. list
    if isinstance(value, list):
        n = len(value)
        width = len(str(n)) if n > 0 else 1
        return [
            _dump_value(
                item,
                path=path.index(i, width=width),
                bundle_dir=bundle_dir,
                registry=registry,
            )
            for i, item in enumerate(value)
        ]

    # 3. dict with safe string keys
    if isinstance(value, dict):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Unsupported result value at {path.display()}:\n"
                    f"dict result keys must be strings in Stage 1; "
                    f"got {type(key).__name__} key {key!r}."
                )
            if key == _FURU_KEY:
                raise ValueError(
                    f"Unsupported result value at {path.display()}:\n"
                    f"dict keys named '{_FURU_KEY}' are reserved by Furu result persistence."
                )
            if not _is_safe_path_segment(key):
                raise ValueError(
                    f"Unsupported result path at {path.key(key).display()}:\n"
                    "dict key cannot be used as an artifact path segment."
                )
            out[key] = _dump_value(
                item,
                path=path.key(key),
                bundle_dir=bundle_dir,
                registry=registry,
            )
        return out

    # 4. dataclass instance
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        type_id = fully_qualified_name(type(value))
        fields_out: dict[str, JsonValue] = {}
        for field in dataclasses.fields(value):
            name = field.name
            if not _is_safe_path_segment(name):
                raise ValueError(
                    f"Unsupported result path at {path.field(name).display()}:\n"
                    "dataclass field name cannot be used as an artifact path segment."
                )
            fields_out[name] = _dump_value(
                getattr(value, name),
                path=path.field(name),
                bundle_dir=bundle_dir,
                registry=registry,
            )
        return {
            _FURU_KEY: {
                "kind": "dataclass",
                "type": type_id,
                "fields": fields_out,
            }
        }

    # 5. Pydantic BaseModel
    if isinstance(value, PydanticBaseModel):
        type_id = fully_qualified_name(type(value))
        fields_out_p: dict[str, JsonValue] = {}
        for name in type(value).model_fields:
            if not _is_safe_path_segment(name):
                raise ValueError(
                    f"Unsupported result path at {path.field(name).display()}:\n"
                    "pydantic field name cannot be used as an artifact path segment."
                )
            fields_out_p[name] = _dump_value(
                getattr(value, name),
                path=path.field(name),
                bundle_dir=bundle_dir,
                registry=registry,
            )
        return {
            _FURU_KEY: {
                "kind": "pydantic",
                "type": type_id,
                "fields": fields_out_p,
            }
        }

    # 6. registered built-in external codec
    codec = registry.codec_for_value(value)
    if codec is not None:
        artifact_subdir = path.artifact_dir()
        artifact_dir = bundle_dir / artifact_subdir
        meta = codec.dump(value, artifact_dir=artifact_dir, path=path)
        return {
            _FURU_KEY: {
                "kind": "external",
                "codec": codec.codec_id,
                "path": str(artifact_subdir).replace("\\", "/"),
                "meta": meta,
            }
        }

    # 7. unsupported
    raise ValueError(
        f"Unsupported result value at {path.display()}:\n"
        f"value of type {type(value).__name__!r} is not supported by Furu result persistence."
    )


def dump_result_tree(
    value: object,
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    return _dump_value(
        value,
        path=LogicalPath(),
        bundle_dir=bundle_dir,
        registry=registry,
    )


def _validate_external_artifact_path(
    *,
    bundle_dir: Path,
    artifact_relpath: str,
) -> Path:
    if not isinstance(artifact_relpath, str):
        raise ValueError(
            f"external artifact path must be a string, got {type(artifact_relpath).__name__}"
        )
    rel = Path(artifact_relpath)
    if rel.is_absolute():
        raise ValueError(
            f"external artifact path must be relative, got {artifact_relpath!r}"
        )
    parts = rel.parts
    if not parts or parts[0] != _ARTIFACTS_DIRNAME:
        raise ValueError(
            f"external artifact path must start with {_ARTIFACTS_DIRNAME!r}/, got {artifact_relpath!r}"
        )
    if any(part == ".." for part in parts):
        raise ValueError(
            f"external artifact path must not contain '..', got {artifact_relpath!r}"
        )

    bundle_resolved = bundle_dir.resolve()
    artifact_dir = (bundle_dir / rel).resolve()
    try:
        artifact_dir.relative_to(bundle_resolved)
    except ValueError as exc:
        raise ValueError(
            f"external artifact path escapes bundle directory: {artifact_relpath!r}"
        ) from exc

    if not artifact_dir.exists():
        raise ValueError(
            f"external artifact directory does not exist: {artifact_dir}"
        )
    return artifact_dir


def _load_value(
    node: JsonValue,
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    if _is_json_scalar(node):
        return node

    if isinstance(node, list):
        return [
            _load_value(item, bundle_dir=bundle_dir, registry=registry)
            for item in node
        ]

    if isinstance(node, dict):
        if _is_furu_wrapper(node):
            wrapper = node[_FURU_KEY]
            if not isinstance(wrapper, dict):
                raise ValueError(
                    f"invalid Furu wrapper: '{_FURU_KEY}' value must be an object"
                )
            kind = wrapper.get("kind")
            if kind == "external":
                codec_id = wrapper.get("codec")
                if not isinstance(codec_id, str):
                    raise ValueError(
                        "invalid external Furu wrapper: 'codec' must be a string"
                    )
                codec = registry.codec_by_id(codec_id)
                rel_path = wrapper.get("path")
                if not isinstance(rel_path, str):
                    raise ValueError(
                        "invalid external Furu wrapper: 'path' must be a string"
                    )
                artifact_dir = _validate_external_artifact_path(
                    bundle_dir=bundle_dir,
                    artifact_relpath=rel_path,
                )
                meta = wrapper.get("meta")
                return codec.load(artifact_dir=artifact_dir, meta=meta)

            if kind == "dataclass":
                type_id = wrapper.get("type")
                if not isinstance(type_id, str):
                    raise ValueError(
                        "invalid dataclass Furu wrapper: 'type' must be a string"
                    )
                cls = _resolve_type_id(type_id)
                if not dataclasses.is_dataclass(cls):
                    raise ValueError(
                        f"resolved type {type_id!r} is not a dataclass"
                    )
                fields_node = wrapper.get("fields")
                if not isinstance(fields_node, dict):
                    raise ValueError(
                        "invalid dataclass Furu wrapper: 'fields' must be an object"
                    )
                loaded: dict[str, object] = {}
                for name, field_node in fields_node.items():
                    loaded[name] = _load_value(
                        field_node,
                        bundle_dir=bundle_dir,
                        registry=registry,
                    )
                obj = object.__new__(cls)
                for name, val in loaded.items():
                    object.__setattr__(obj, name, val)
                return obj

            if kind == "pydantic":
                type_id = wrapper.get("type")
                if not isinstance(type_id, str):
                    raise ValueError(
                        "invalid pydantic Furu wrapper: 'type' must be a string"
                    )
                cls = _resolve_type_id(type_id)
                if not (isinstance(cls, type) and issubclass(cls, PydanticBaseModel)):
                    raise ValueError(
                        f"resolved type {type_id!r} is not a Pydantic BaseModel subclass"
                    )
                fields_node = wrapper.get("fields")
                if not isinstance(fields_node, dict):
                    raise ValueError(
                        "invalid pydantic Furu wrapper: 'fields' must be an object"
                    )
                loaded_fields: dict[str, object] = {}
                for name, field_node in fields_node.items():
                    loaded_fields[name] = _load_value(
                        field_node,
                        bundle_dir=bundle_dir,
                        registry=registry,
                    )
                return cls.model_construct(**loaded_fields)

            raise ValueError(f"unknown Furu wrapper kind: {kind!r}")

        out: dict[str, object] = {}
        for key, val in node.items():
            out[key] = _load_value(val, bundle_dir=bundle_dir, registry=registry)
        return out

    raise ValueError(f"unsupported JSON node type: {type(node).__name__}")


def load_result_tree(
    node: JsonValue,
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    return _load_value(node, bundle_dir=bundle_dir, registry=registry)


def _write_manifest_json(path: Path, manifest: Any) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def save_result_bundle(value: object, bundle_dir: Path) -> None:
    if bundle_dir.exists():
        raise FileExistsError(bundle_dir)

    bundle_dir.mkdir(parents=True)
    (bundle_dir / _ARTIFACTS_DIRNAME).mkdir()

    registry = default_result_registry()
    manifest = dump_result_tree(
        value,
        bundle_dir=bundle_dir,
        registry=registry,
    )

    _write_manifest_json(bundle_dir / _MANIFEST_NAME, manifest)


def load_result_bundle(bundle_dir: Path) -> object:
    manifest_path = bundle_dir / _MANIFEST_NAME

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    registry = default_result_registry()

    return load_result_tree(
        raw,
        bundle_dir=bundle_dir,
        registry=registry,
    )


def result_bundle_is_complete(bundle_dir: Path) -> bool:
    return (bundle_dir / _MANIFEST_NAME).exists()
