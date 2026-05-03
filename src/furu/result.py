from __future__ import annotations

import dataclasses
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel

type JsonValue = (
    list[JsonValue] | dict[str, JsonValue] | str | bool | int | float | None
)

_RESERVED_KEY = "$furu"


@dataclass(frozen=True)
class _PathKey:
    value: str


@dataclass(frozen=True)
class _PathIndex:
    value: str


@dataclass(frozen=True)
class LogicalPath:
    parts: tuple[object, ...] = ()

    def key(self, key: str) -> LogicalPath:
        return LogicalPath((*self.parts, _PathKey(key)))

    def index(self, index: int, *, width: int) -> LogicalPath:
        return LogicalPath((*self.parts, _PathIndex(f"{index:0{width}d}")))

    def field(self, name: str) -> LogicalPath:
        return self.key(name)

    def display(self) -> str:
        output = "$"
        for part in self.parts:
            if isinstance(part, _PathIndex):
                output += f"[{part.value}]"
            elif isinstance(part, _PathKey):
                if part.value.isidentifier():
                    output += f".{part.value}"
                else:
                    output += f"[{json.dumps(part.value)}]"
            else:
                raise TypeError(f"unexpected logical path part {part!r}")
        return output

    def artifact_dir(self) -> Path:
        if not self.parts:
            return Path("artifacts") / "root"

        path = Path("artifacts")
        for part in self.parts:
            if isinstance(part, (_PathKey, _PathIndex)):
                path /= part.value
            else:
                raise TypeError(f"unexpected logical path part {part!r}")
        return path


class ResultCodec(Protocol):
    codec_id: str

    def matches(self, value: object) -> bool: ...

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

    def matches(self, value: object) -> bool:
        return isinstance(value, self._np.ndarray)

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        array = cast(Any, value)
        if array.dtype.hasobject:
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                "numpy object-dtype arrays are not supported by the default npy codec."
            )

        artifact_dir.mkdir(parents=True, exist_ok=False)
        self._np.save(artifact_dir / "data.npy", array, allow_pickle=False)
        return {
            "shape": [int(dim) for dim in array.shape],
            "dtype": str(array.dtype),
        }

    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object:
        return self._np.load(artifact_dir / "data.npy", allow_pickle=False)


class PolarsParquetCodec:
    codec_id = "polars.dataframe.parquet"

    def __init__(self) -> None:
        self._pl: Any = importlib.import_module("polars")

    def matches(self, value: object) -> bool:
        return isinstance(value, self._pl.DataFrame)

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        del path
        frame = cast(Any, value)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        frame.write_parquet(artifact_dir / "data.parquet")
        return {
            "rows": int(frame.height),
            "columns": list(frame.columns),
        }

    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object:
        del meta
        return self._pl.read_parquet(artifact_dir / "data.parquet")


class ResultRegistry:
    def __init__(self) -> None:
        self._codecs: list[ResultCodec] = []
        self._by_id: dict[str, ResultCodec] = {}

    def register(self, codec: ResultCodec) -> None:
        if codec.codec_id in self._by_id:
            raise ValueError(f"duplicate result codec id {codec.codec_id!r}")
        self._codecs.append(codec)
        self._by_id[codec.codec_id] = codec

    def codec_for_value(self, value: object) -> ResultCodec | None:
        for codec in self._codecs:
            if codec.matches(value):
                return codec
        return None

    def codec_by_id(self, codec_id: str) -> ResultCodec:
        try:
            return self._by_id[codec_id]
        except KeyError:
            raise ValueError(f"Unknown result codec {codec_id!r}.") from None


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


def _dump_value(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    if value is None or type(value) in (bool, int, float, str):
        return cast(JsonValue, value)

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
        return _dump_dict(
            cast(dict[object, object], value),
            path=path,
            bundle_dir=bundle_dir,
            registry=registry,
        )

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dump_dataclass(
            value,
            path=path,
            bundle_dir=bundle_dir,
            registry=registry,
        )

    if isinstance(value, BaseModel):
        return _dump_pydantic(
            value,
            path=path,
            bundle_dir=bundle_dir,
            registry=registry,
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
            _RESERVED_KEY: {
                "kind": "external",
                "codec": codec.codec_id,
                "path": artifact_path.as_posix(),
                "meta": meta,
            }
        }

    raise ValueError(
        f"Unsupported result value at {path.display()}:\n"
        f"values of type {type(value).__module__}.{type(value).__qualname__} "
        "are not supported by Furu result persistence in Stage 1."
    )


def _dump_dict(
    value: dict[object, object],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                "dict result keys must be strings in Stage 1; "
                f"got {type(key).__name__} key {key!r}."
            )
        if key == _RESERVED_KEY:
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                "dict keys named '$furu' are reserved by Furu result persistence."
            )
        child_path = path.key(key)
        _validate_path_segment(
            key,
            path=child_path,
            description="dict key",
        )
        output[key] = _dump_value(
            item,
            path=child_path,
            bundle_dir=bundle_dir,
            registry=registry,
        )
    return output


def _dump_dataclass(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> dict[str, JsonValue]:
    cls = type(value)
    type_id = _fully_qualified_result_type(
        cls,
        path=path,
        kind="dataclass",
    )
    fields: dict[str, JsonValue] = {}
    for field in dataclasses.fields(cast(Any, value)):
        child_path = path.field(field.name)
        _validate_path_segment(
            field.name,
            path=child_path,
            description="dataclass field name",
        )
        fields[field.name] = _dump_value(
            getattr(value, field.name),
            path=child_path,
            bundle_dir=bundle_dir,
            registry=registry,
        )

    return {
        _RESERVED_KEY: {
            "kind": "dataclass",
            "type": type_id,
            "fields": fields,
        }
    }


def _dump_pydantic(
    value: BaseModel,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> dict[str, JsonValue]:
    cls = type(value)
    type_id = _fully_qualified_result_type(
        cls,
        path=path,
        kind="pydantic",
    )
    fields: dict[str, JsonValue] = {}
    for name in value.__class__.model_fields:
        child_path = path.field(name)
        _validate_path_segment(
            name,
            path=child_path,
            description="pydantic field name",
        )
        fields[name] = _dump_value(
            getattr(value, name),
            path=child_path,
            bundle_dir=bundle_dir,
            registry=registry,
        )

    return {
        _RESERVED_KEY: {
            "kind": "pydantic",
            "type": type_id,
            "fields": fields,
        }
    }


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


def _load_value(
    node: JsonValue,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    if node is None or type(node) in (bool, int, float, str):
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
        if _RESERVED_KEY in node:
            return _load_wrapper(
                node,
                path=path,
                bundle_dir=bundle_dir,
                registry=registry,
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

    raise ValueError(f"Unsupported manifest node at {path.display()}.")


def _load_wrapper(
    node: dict[str, JsonValue],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    if set(node) != {_RESERVED_KEY}:
        raise ValueError(
            f"Unsupported result wrapper at {path.display()}:\n"
            "Furu wrapper objects may only contain the '$furu' key."
        )
    wrapper = node[_RESERVED_KEY]
    if not isinstance(wrapper, dict):
        raise ValueError(
            f"Unsupported result wrapper at {path.display()}:\n"
            "the '$furu' value must be an object."
        )
    kind = wrapper.get("kind")
    if kind == "external":
        return _load_external_wrapper(
            wrapper,
            path=path,
            bundle_dir=bundle_dir,
            registry=registry,
        )
    if kind == "dataclass":
        return _load_dataclass_wrapper(
            wrapper,
            path=path,
            bundle_dir=bundle_dir,
            registry=registry,
        )
    if kind == "pydantic":
        return _load_pydantic_wrapper(
            wrapper,
            path=path,
            bundle_dir=bundle_dir,
            registry=registry,
        )
    raise ValueError(
        f"Unsupported result wrapper at {path.display()}:\n"
        f"unknown Furu wrapper kind {kind!r}."
    )


def _load_external_wrapper(
    wrapper: dict[str, JsonValue],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    codec_id = wrapper.get("codec")
    artifact_path_text = wrapper.get("path")
    meta = wrapper.get("meta")
    if not isinstance(codec_id, str):
        raise ValueError(
            f"Unsupported external result wrapper at {path.display()}:\n"
            "external wrapper codec must be a string."
        )
    if not isinstance(artifact_path_text, str):
        raise ValueError(
            f"Unsupported external result wrapper at {path.display()}:\n"
            "external wrapper path must be a string."
        )

    artifact_dir = _validated_artifact_dir(
        artifact_path_text,
        bundle_dir=bundle_dir,
        path=path,
    )
    codec = registry.codec_by_id(codec_id)
    return codec.load(artifact_dir=artifact_dir, meta=meta)


def _load_dataclass_wrapper(
    wrapper: dict[str, JsonValue],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    type_id = wrapper.get("type")
    field_nodes = wrapper.get("fields")
    if not isinstance(type_id, str):
        raise ValueError(
            f"Unsupported dataclass result wrapper at {path.display()}:\n"
            "dataclass wrapper type must be a string."
        )
    if not isinstance(field_nodes, dict):
        raise ValueError(
            f"Unsupported dataclass result wrapper at {path.display()}:\n"
            "dataclass wrapper fields must be an object."
        )

    cls = _resolve_type(type_id, path=path)
    if not dataclasses.is_dataclass(cls):
        raise ValueError(
            f"Unsupported dataclass result wrapper at {path.display()}:\n"
            f"{type_id!r} does not resolve to a dataclass type."
        )

    loaded_fields = {
        name: _load_value(
            field_node,
            path=path.field(name),
            bundle_dir=bundle_dir,
            registry=registry,
        )
        for name, field_node in field_nodes.items()
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
    type_id = wrapper.get("type")
    field_nodes = wrapper.get("fields")
    if not isinstance(type_id, str):
        raise ValueError(
            f"Unsupported pydantic result wrapper at {path.display()}:\n"
            "pydantic wrapper type must be a string."
        )
    if not isinstance(field_nodes, dict):
        raise ValueError(
            f"Unsupported pydantic result wrapper at {path.display()}:\n"
            "pydantic wrapper fields must be an object."
        )

    cls = _resolve_type(type_id, path=path)
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise ValueError(
            f"Unsupported pydantic result wrapper at {path.display()}:\n"
            f"{type_id!r} does not resolve to a Pydantic model type."
        )

    loaded_fields = {
        name: _load_value(
            field_node,
            path=path.field(name),
            bundle_dir=bundle_dir,
            registry=registry,
        )
        for name, field_node in field_nodes.items()
    }
    model_cls = cast(Any, cls)
    return model_cls.model_construct(**loaded_fields)


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
    return load_result_tree(raw, bundle_dir=bundle_dir, registry=registry)


def result_bundle_is_complete(bundle_dir: Path) -> bool:
    return (bundle_dir / "manifest.json").exists()


def _write_json_file(path: Path, value: JsonValue) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _validate_path_segment(
    segment: str,
    *,
    path: LogicalPath,
    description: str,
) -> None:
    if (
        segment == ""
        or segment == "."
        or segment == ".."
        or "/" in segment
        or "\\" in segment
        or "\x00" in segment
    ):
        raise ValueError(
            f"Unsupported result path at {path.display()}:\n"
            f"{description} cannot be used as an artifact path segment."
        )


def _fully_qualified_result_type(
    cls: type,
    *,
    path: LogicalPath,
    kind: str,
) -> str:
    module = cls.__module__
    qualname = cls.__qualname__
    if module == "__main__" or "<locals>" in qualname or "." in qualname:
        raise ValueError(
            f"Unsupported result value at {path.display()}:\n"
            f"only importable {kind} types are supported by Furu result persistence; "
            f"got {module}.{qualname}."
        )
    return f"{module}.{qualname}"


def _resolve_type(type_id: str, *, path: LogicalPath) -> type:
    module_name, separator, qualname = type_id.rpartition(".")
    if not separator or not module_name or not qualname:
        raise ValueError(
            f"Unsupported result wrapper at {path.display()}:\n"
            f"invalid type id {type_id!r}."
        )
    try:
        module = importlib.import_module(module_name)
        resolved = getattr(module, qualname)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Unsupported result wrapper at {path.display()}:\n"
            f"could not resolve type id {type_id!r}."
        ) from exc
    if not isinstance(resolved, type):
        raise ValueError(
            f"Unsupported result wrapper at {path.display()}:\n"
            f"type id {type_id!r} did not resolve to a type."
        )
    return resolved


def _validated_artifact_dir(
    artifact_path_text: str,
    *,
    bundle_dir: Path,
    path: LogicalPath,
) -> Path:
    artifact_path = Path(artifact_path_text)
    if artifact_path.is_absolute():
        raise ValueError(
            f"Unsupported external result wrapper at {path.display()}:\n"
            "external artifact path must be relative."
        )
    if len(artifact_path.parts) < 2 or artifact_path.parts[0] != "artifacts":
        raise ValueError(
            f"Unsupported external result wrapper at {path.display()}:\n"
            "external artifact path must start with 'artifacts/'."
        )

    bundle_root = bundle_dir.resolve()
    artifact_root = (bundle_dir / "artifacts").resolve()
    artifact_dir = (bundle_dir / artifact_path).resolve()
    if not artifact_dir.is_relative_to(bundle_root) or not artifact_dir.is_relative_to(
        artifact_root
    ):
        raise ValueError(
            f"Unsupported external result wrapper at {path.display()}:\n"
            "external artifact path must not escape the result bundle."
        )
    if not artifact_dir.is_dir():
        raise ValueError(
            f"Unsupported external result wrapper at {path.display()}:\n"
            f"external artifact directory does not exist: {artifact_path_text}."
        )
    return artifact_dir
