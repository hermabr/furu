from __future__ import annotations

import dataclasses
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel

from furu.utils import JsonValue

FURU_WRAPPER_KEY = "$furu"


@dataclass(frozen=True)
class _KeyPart:
    key: str


@dataclass(frozen=True)
class _IndexPart:
    index: int
    segment: str


type _PathPart = _KeyPart | _IndexPart


@dataclass(frozen=True)
class LogicalPath:
    parts: tuple[_PathPart, ...] = ()

    def key(self, key: str) -> LogicalPath:
        return LogicalPath((*self.parts, _KeyPart(key)))

    def index(self, index: int, *, width: int) -> LogicalPath:
        return LogicalPath((*self.parts, _IndexPart(index, f"{index:0{width}d}")))

    def field(self, name: str) -> LogicalPath:
        return self.key(name)

    def display(self) -> str:
        out = "$"
        for part in self.parts:
            match part:
                case _KeyPart(key):
                    if _can_display_as_attribute(key):
                        out += f".{key}"
                    else:
                        out += f"[{json.dumps(key)}]"
                case _IndexPart(segment=segment):
                    out += f"[{segment}]"
        return out

    def artifact_dir(self) -> Path:
        if not self.parts:
            return Path("artifacts") / "root"

        path = Path("artifacts")
        for part in self.parts:
            match part:
                case _KeyPart(key):
                    path /= key
                case _IndexPart(segment=segment):
                    path /= segment
        return path


class ResultCodec(Protocol):
    codec_id: str

    def can_dump(self, value: object) -> bool: ...

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue: ...

    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object: ...


class NumpyNpyCodec:
    codec_id = "numpy.ndarray.npy"

    def __init__(self) -> None:
        self._np: Any = importlib.import_module("numpy")

    def can_dump(self, value: object) -> bool:
        return isinstance(value, self._np.ndarray)

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        if not isinstance(value, self._np.ndarray):
            raise TypeError(type(value).__name__)
        if value.dtype.hasobject:
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                "numpy object-dtype arrays are not supported by the default npy codec."
            )

        artifact_dir.mkdir(parents=True, exist_ok=False)
        self._np.save(artifact_dir / "data.npy", value, allow_pickle=False)
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object:
        return self._np.load(artifact_dir / "data.npy", allow_pickle=False)


class PolarsParquetCodec:
    codec_id = "polars.dataframe.parquet"

    def __init__(self) -> None:
        self._pl: Any = importlib.import_module("polars")

    def can_dump(self, value: object) -> bool:
        return isinstance(value, self._pl.DataFrame)

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        if not isinstance(value, self._pl.DataFrame):
            raise TypeError(type(value).__name__)

        artifact_dir.mkdir(parents=True, exist_ok=False)
        value.write_parquet(artifact_dir / "data.parquet")
        return {
            "rows": value.height,
            "columns": list(value.columns),
        }

    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object:
        return self._pl.read_parquet(artifact_dir / "data.parquet")


class ResultRegistry:
    def __init__(self) -> None:
        self._codecs: list[ResultCodec] = []
        self._by_id: dict[str, ResultCodec] = {}

    def register(self, codec: ResultCodec) -> None:
        if codec.codec_id in self._by_id:
            raise ValueError(f"duplicate result codec id: {codec.codec_id}")
        self._codecs.append(codec)
        self._by_id[codec.codec_id] = codec

    def codec_for_value(self, value: object) -> ResultCodec | None:
        for codec in self._codecs:
            if codec.can_dump(value):
                return codec
        return None

    def codec_by_id(self, codec_id: str) -> ResultCodec:
        try:
            return self._by_id[codec_id]
        except KeyError as exc:
            raise ValueError(f"unknown result codec id: {codec_id}") from exc


def default_result_registry() -> ResultRegistry:
    registry = ResultRegistry()

    try:
        registry.register(NumpyNpyCodec())
    except ImportError:
        pass

    try:
        registry.register(PolarsParquetCodec())
    except ImportError:
        pass

    return registry


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


def load_result_tree(
    node: JsonValue,
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    return _load_value(
        node,
        path=LogicalPath(),
        bundle_dir=bundle_dir,
        registry=registry,
    )


def save_result_bundle(value: object, bundle_dir: Path) -> None:
    if bundle_dir.exists():
        raise FileExistsError(bundle_dir)

    bundle_dir.mkdir(parents=True)
    (bundle_dir / "artifacts").mkdir()

    registry = default_result_registry()
    manifest = dump_result_tree(
        value,
        bundle_dir=bundle_dir,
        registry=registry,
    )

    _write_json_file(bundle_dir / "manifest.json", manifest)


def load_result_bundle(bundle_dir: Path) -> object:
    manifest_path = bundle_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = default_result_registry()
    return load_result_tree(
        raw,
        bundle_dir=bundle_dir,
        registry=registry,
    )


def result_bundle_is_complete(bundle_dir: Path) -> bool:
    return (bundle_dir / "manifest.json").exists()


def _dump_value(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if isinstance(value, list):
        width = len(str(len(value)))
        return [
            _dump_value(
                item,
                path=path.index(index, width=width),
                bundle_dir=bundle_dir,
                registry=registry,
            )
            for index, item in enumerate(value)
        ]

    if isinstance(value, dict):
        return _dump_dict(value, path=path, bundle_dir=bundle_dir, registry=registry)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dump_dataclass(
            value, path=path, bundle_dir=bundle_dir, registry=registry
        )

    if isinstance(value, BaseModel):
        return _dump_pydantic(
            value, path=path, bundle_dir=bundle_dir, registry=registry
        )

    codec = registry.codec_for_value(value)
    if codec is not None:
        artifact_path = path.artifact_dir()
        meta = codec.dump(
            value,
            artifact_dir=bundle_dir / artifact_path,
            path=path,
        )
        return {
            FURU_WRAPPER_KEY: {
                "kind": "external",
                "codec": codec.codec_id,
                "path": artifact_path.as_posix(),
                "meta": meta,
            }
        }

    raise ValueError(
        f"Unsupported result value at {path.display()}:\n"
        f"{type(value).__module__}.{type(value).__qualname__} is not supported "
        "by Furu result persistence."
    )


def _dump_dict(
    value: dict[Any, Any],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    dumped: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                "dict result keys must be strings in Stage 1; "
                f"got {type(key).__name__} key {key!r}."
            )
        _validate_dict_key(key, parent_path=path)
        dumped[key] = _dump_value(
            item,
            path=path.key(key),
            bundle_dir=bundle_dir,
            registry=registry,
        )
    return dumped


def _dump_dataclass(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    cls = type(value)
    fields: dict[str, JsonValue] = {}
    for field in dataclasses.fields(cast(Any, value)):
        _validate_field_name(field.name, path=path)
        fields[field.name] = _dump_value(
            getattr(value, field.name),
            path=path.field(field.name),
            bundle_dir=bundle_dir,
            registry=registry,
        )

    return {
        FURU_WRAPPER_KEY: {
            "kind": "dataclass",
            "type": _fully_qualified_importable_type(cls, path=path),
            "fields": fields,
        }
    }


def _dump_pydantic(
    value: BaseModel,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    cls = type(value)
    fields: dict[str, JsonValue] = {}
    for name in cls.model_fields:
        _validate_field_name(name, path=path)
        fields[name] = _dump_value(
            getattr(value, name),
            path=path.field(name),
            bundle_dir=bundle_dir,
            registry=registry,
        )

    return {
        FURU_WRAPPER_KEY: {
            "kind": "pydantic",
            "type": _fully_qualified_importable_type(cls, path=path),
            "fields": fields,
        }
    }


def _load_value(
    node: JsonValue,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    if node is None or isinstance(node, bool | int | float | str):
        return node

    if isinstance(node, list):
        width = len(str(len(node)))
        return [
            _load_value(
                item,
                path=path.index(index, width=width),
                bundle_dir=bundle_dir,
                registry=registry,
            )
            for index, item in enumerate(node)
        ]

    if isinstance(node, dict):
        if FURU_WRAPPER_KEY in node:
            return _load_wrapper(
                node, path=path, bundle_dir=bundle_dir, registry=registry
            )
        return {
            key: _load_value(
                item,
                path=path.key(key),
                bundle_dir=bundle_dir,
                registry=registry,
            )
            for key, item in node.items()
        }

    raise TypeError(type(node).__name__)


def _load_wrapper(
    node: dict[str, JsonValue],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    if set(node) != {FURU_WRAPPER_KEY}:
        raise ValueError(f"Invalid Furu wrapper at {path.display()}: extra keys found.")

    wrapper = node[FURU_WRAPPER_KEY]
    if not isinstance(wrapper, dict):
        raise ValueError(f"Invalid Furu wrapper at {path.display()}: expected object.")

    kind = wrapper.get("kind")
    match kind:
        case "external":
            return _load_external_wrapper(
                cast(dict[str, JsonValue], wrapper),
                path=path,
                bundle_dir=bundle_dir,
                registry=registry,
            )
        case "dataclass":
            return _load_dataclass_wrapper(
                cast(dict[str, JsonValue], wrapper),
                path=path,
                bundle_dir=bundle_dir,
                registry=registry,
            )
        case "pydantic":
            return _load_pydantic_wrapper(
                cast(dict[str, JsonValue], wrapper),
                path=path,
                bundle_dir=bundle_dir,
                registry=registry,
            )
        case _:
            raise ValueError(
                f"Invalid Furu wrapper at {path.display()}: unknown kind {kind!r}."
            )


def _load_external_wrapper(
    wrapper: dict[str, JsonValue],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    codec_id = wrapper.get("codec")
    if not isinstance(codec_id, str):
        raise ValueError(f"Invalid external result wrapper at {path.display()}: codec.")

    path_text = wrapper.get("path")
    if not isinstance(path_text, str):
        raise ValueError(f"Invalid external result wrapper at {path.display()}: path.")

    artifact_path = Path(path_text)
    artifact_dir = _validated_artifact_dir(
        artifact_path,
        bundle_dir=bundle_dir,
        path=path,
    )
    meta = wrapper.get("meta")
    codec = registry.codec_by_id(codec_id)
    return codec.load(artifact_dir=artifact_dir, meta=meta)


def _load_dataclass_wrapper(
    wrapper: dict[str, JsonValue],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    cls = _load_wrapper_type(wrapper, path=path)
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"Resolved type at {path.display()} is not a dataclass.")

    raw_fields = wrapper.get("fields")
    if not isinstance(raw_fields, dict):
        raise ValueError(f"Invalid dataclass wrapper at {path.display()}: fields.")

    loaded_fields = {
        name: _load_value(
            value,
            path=path.field(name),
            bundle_dir=bundle_dir,
            registry=registry,
        )
        for name, value in raw_fields.items()
    }
    obj = object.__new__(cls)
    for name, value in loaded_fields.items():
        object.__setattr__(obj, name, value)
    return obj


def _load_pydantic_wrapper(
    wrapper: dict[str, JsonValue],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    cls = _load_wrapper_type(wrapper, path=path)
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise ValueError(f"Resolved type at {path.display()} is not a Pydantic model.")

    raw_fields = wrapper.get("fields")
    if not isinstance(raw_fields, dict):
        raise ValueError(f"Invalid Pydantic wrapper at {path.display()}: fields.")

    loaded_fields = {
        name: _load_value(
            value,
            path=path.field(name),
            bundle_dir=bundle_dir,
            registry=registry,
        )
        for name, value in raw_fields.items()
    }
    return cls.model_construct(_fields_set=set(loaded_fields), **loaded_fields)


def _load_wrapper_type(wrapper: dict[str, JsonValue], *, path: LogicalPath) -> type:
    type_id = wrapper.get("type")
    if not isinstance(type_id, str):
        raise ValueError(f"Invalid Furu wrapper at {path.display()}: type.")
    resolved = _resolve_type(type_id)
    if not isinstance(resolved, type):
        raise ValueError(f"Resolved object at {path.display()} is not a type.")
    return resolved


def _resolve_type(type_id: str) -> object:
    module_name, _, qualname = type_id.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"invalid result type id: {type_id}")
    module = importlib.import_module(module_name)
    return getattr(module, qualname)


def _fully_qualified_importable_type(cls: type, *, path: LogicalPath) -> str:
    if cls.__module__ == "__main__":
        raise ValueError(
            f"Unsupported result value at {path.display()}:\n"
            "result dataclass and Pydantic model types must be importable; "
            "__main__ classes are not supported."
        )
    if "<locals>" in cls.__qualname__:
        raise ValueError(
            f"Unsupported result value at {path.display()}:\n"
            "result dataclass and Pydantic model types must be importable; "
            "local classes are not supported."
        )
    if "." in cls.__qualname__:
        raise ValueError(
            f"Unsupported result value at {path.display()}:\n"
            "result dataclass and Pydantic model types must be importable; "
            "nested classes are not supported."
        )

    type_id = f"{cls.__module__}.{cls.__qualname__}"
    try:
        resolved = _resolve_type(type_id)
    except (AttributeError, ImportError, ValueError) as exc:
        raise ValueError(
            f"Unsupported result value at {path.display()}:\n"
            f"{type_id} is not an importable result type."
        ) from exc
    if resolved is not cls:
        raise ValueError(
            f"Unsupported result value at {path.display()}:\n"
            f"{type_id} does not resolve to the result type."
        )
    return type_id


def _validated_artifact_dir(
    artifact_path: Path,
    *,
    bundle_dir: Path,
    path: LogicalPath,
) -> Path:
    if artifact_path.is_absolute():
        raise ValueError(
            f"Invalid external artifact path at {path.display()}: absolute."
        )
    if len(artifact_path.parts) < 2 or artifact_path.parts[0] != "artifacts":
        raise ValueError(
            f"Invalid external artifact path at {path.display()}: "
            "path must start with artifacts/."
        )

    artifact_dir = bundle_dir / artifact_path
    bundle_root = bundle_dir.resolve(strict=True)
    artifacts_root = (bundle_dir / "artifacts").resolve(strict=True)
    resolved_artifact_dir = artifact_dir.resolve(strict=False)
    try:
        resolved_artifact_dir.relative_to(bundle_root)
        resolved_artifact_dir.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(
            f"Invalid external artifact path at {path.display()}: "
            "path escapes the result bundle."
        ) from exc
    if not artifact_dir.exists():
        raise FileNotFoundError(artifact_dir)
    if not artifact_dir.is_dir():
        raise NotADirectoryError(artifact_dir)
    return artifact_dir


def _validate_dict_key(key: str, *, parent_path: LogicalPath) -> None:
    if key == FURU_WRAPPER_KEY:
        raise ValueError(
            f"Unsupported result value at {parent_path.display()}:\n"
            "dict keys named '$furu' are reserved by Furu result persistence."
        )
    if not _is_safe_path_segment(key):
        raise ValueError(
            f"Unsupported result path at {parent_path.key(key).display()}:\n"
            "dict key cannot be used as an artifact path segment."
        )


def _validate_field_name(name: str, *, path: LogicalPath) -> None:
    if name == FURU_WRAPPER_KEY:
        raise ValueError(
            f"Unsupported result value at {path.display()}:\n"
            "field names named '$furu' are reserved by Furu result persistence."
        )
    if not _is_safe_path_segment(name):
        raise ValueError(
            f"Unsupported result path at {path.field(name).display()}:\n"
            "field name cannot be used as an artifact path segment."
        )


def _is_safe_path_segment(value: str) -> bool:
    return (
        value not in {"", ".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\0" not in value
    )


def _can_display_as_attribute(value: str) -> bool:
    return value.isidentifier()


def _write_json_file(path: Path, value: JsonValue) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
