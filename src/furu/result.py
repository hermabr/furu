"""Stage 1 result persistence.

Results are stored as a bundle directory containing a JSON manifest and
optional external artifacts:

    result/
        manifest.json
        artifacts/
            <logical path>/
                data.npy

The manifest is the completion marker. A partial bundle without
``manifest.json`` is not considered a valid cached result.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

import pydantic

from furu.utils import JsonValue, fully_qualified_name

WRAPPER_KEY = "$furu"
ARTIFACTS_DIR_NAME = "artifacts"
MANIFEST_FILE_NAME = "manifest.json"
_ROOT_ARTIFACT_NAME = "root"


@dataclass(frozen=True)
class _PathSegment:
    artifact: str
    display: str


@dataclass(frozen=True)
class LogicalPath:
    parts: tuple[_PathSegment, ...] = ()

    def key(self, key: str) -> LogicalPath:
        display = f".{key}" if _is_simple_identifier(key) else f"[{json.dumps(key)}]"
        return LogicalPath(self.parts + (_PathSegment(key, display),))

    def field(self, name: str) -> LogicalPath:
        return self.key(name)

    def index(self, index: int, *, width: int) -> LogicalPath:
        segment = f"{index:0{width}d}"
        return LogicalPath(self.parts + (_PathSegment(segment, f"[{segment}]"),))

    def display(self) -> str:
        return "$" + "".join(part.display for part in self.parts)

    def artifact_dir(self) -> Path:
        if not self.parts:
            return Path(ARTIFACTS_DIR_NAME) / _ROOT_ARTIFACT_NAME
        return Path(ARTIFACTS_DIR_NAME, *(part.artifact for part in self.parts))


def _is_simple_identifier(text: str) -> bool:
    return bool(text) and text.isidentifier()


def _is_safe_path_segment(text: str) -> bool:
    return (
        text not in {"", ".", ".."}
        and "/" not in text
        and "\\" not in text
        and "\x00" not in text
    )


class ResultCodec(Protocol):
    codec_id: ClassVar[str]

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
    codec_id: ClassVar[str] = "numpy.ndarray.npy"

    def __init__(self) -> None:
        import numpy as np  # noqa: F401

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
        return {"shape": list(value.shape), "dtype": str(value.dtype)}

    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


@cache
def _codecs() -> tuple[ResultCodec, ...]:
    codecs: list[ResultCodec] = []
    try:
        codecs.append(NumpyNpyCodec())
    except ImportError:
        pass
    return tuple(codecs)


def _codec_for_value(value: object) -> ResultCodec | None:
    for codec in _codecs():
        if codec.matches(value):
            return codec
    return None


def _codec_by_id(codec_id: str) -> ResultCodec:
    for codec in _codecs():
        if codec.codec_id == codec_id:
            return codec
    raise ValueError(f"unknown result codec: {codec_id}")


def dump_result_tree(value: object, *, result_dir: Path) -> JsonValue:
    return _dump_value(value, path=LogicalPath(), result_dir=result_dir)


def _dump_value(
    value: object,
    *,
    path: LogicalPath,
    result_dir: Path,
) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value

    if isinstance(value, list):
        return _dump_sequence(
            cast(list[object], value), path=path, result_dir=result_dir
        )

    if isinstance(value, tuple):
        return {
            WRAPPER_KEY: {
                "kind": "tuple",
                "items": _dump_sequence(list(value), path=path, result_dir=result_dir),
            }
        }

    if isinstance(value, dict):
        return _dump_dict(
            cast(dict[object, object], value), path=path, result_dir=result_dir
        )

    if isinstance(value, pydantic.BaseModel):
        return _dump_pydantic(value, path=path, result_dir=result_dir)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dump_dataclass(value, path=path, result_dir=result_dir)

    codec = _codec_for_value(value)
    if codec is not None:
        return _dump_external(value, codec=codec, path=path, result_dir=result_dir)

    raise ValueError(
        f"Unsupported result value at {path.display()}:\n"
        f"values of type {type(value).__name__!r} are not supported by Furu Stage 1."
    )


def _dump_sequence(
    value: list[object],
    *,
    path: LogicalPath,
    result_dir: Path,
) -> list[JsonValue]:
    width = max(len(str(len(value))), 1)
    return [
        _dump_value(item, path=path.index(i, width=width), result_dir=result_dir)
        for i, item in enumerate(value)
    ]


def _dump_dict(
    value: dict[object, object],
    *,
    path: LogicalPath,
    result_dir: Path,
) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                f"dict result keys must be strings in Stage 1; got {type(key).__name__} key {key!r}."
            )
        if key == WRAPPER_KEY:
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                f"dict keys named {WRAPPER_KEY!r} are reserved by Furu result persistence."
            )
        child_path = path.key(key)
        if not _is_safe_path_segment(key):
            raise ValueError(
                f"Unsupported result path at {child_path.display()}:\n"
                "dict key cannot be used as an artifact path segment."
            )
        out[key] = _dump_value(child, path=child_path, result_dir=result_dir)
    return out


def _dump_dataclass(
    value: object,
    *,
    path: LogicalPath,
    result_dir: Path,
) -> dict[str, JsonValue]:
    fields: dict[str, JsonValue] = {}
    for field in dataclasses.fields(cast(Any, value)):
        if not _is_safe_path_segment(field.name):
            raise ValueError(
                f"Unsupported result path at {path.field(field.name).display()}:\n"
                "dataclass field name cannot be used as an artifact path segment."
            )
        fields[field.name] = _dump_value(
            getattr(value, field.name),
            path=path.field(field.name),
            result_dir=result_dir,
        )
    return {
        WRAPPER_KEY: {
            "kind": "dataclass",
            "type": fully_qualified_name(type(value)),
            "fields": fields,
        }
    }


def _dump_pydantic(
    value: pydantic.BaseModel,
    *,
    path: LogicalPath,
    result_dir: Path,
) -> dict[str, JsonValue]:
    fields: dict[str, JsonValue] = {}
    for name in type(value).model_fields:
        if not _is_safe_path_segment(name):
            raise ValueError(
                f"Unsupported result path at {path.field(name).display()}:\n"
                "pydantic field name cannot be used as an artifact path segment."
            )
        fields[name] = _dump_value(
            getattr(value, name),
            path=path.field(name),
            result_dir=result_dir,
        )
    return {
        WRAPPER_KEY: {
            "kind": "pydantic",
            "type": fully_qualified_name(type(value)),
            "fields": fields,
        }
    }


def _dump_external(
    value: object,
    *,
    codec: ResultCodec,
    path: LogicalPath,
    result_dir: Path,
) -> dict[str, JsonValue]:
    artifact_rel = path.artifact_dir()
    meta = codec.dump(value, artifact_dir=result_dir / artifact_rel, path=path)
    return {
        WRAPPER_KEY: {
            "kind": "external",
            "codec": codec.codec_id,
            "path": artifact_rel.as_posix(),
            "meta": meta,
        }
    }


def load_result_tree(node: JsonValue, *, result_dir: Path) -> object:
    return _load_value(node, result_dir=result_dir)


def _load_value(node: JsonValue, *, result_dir: Path) -> object:
    if node is None or isinstance(node, (bool, int, float, str)):
        return node

    if isinstance(node, list):
        return [_load_value(child, result_dir=result_dir) for child in node]

    if isinstance(node, dict):
        if WRAPPER_KEY in node:
            if set(node) != {WRAPPER_KEY}:
                raise ValueError("malformed Furu wrapper: extra keys found")
            return _load_wrapper(node[WRAPPER_KEY], result_dir=result_dir)
        return {
            key: _load_value(child, result_dir=result_dir)
            for key, child in node.items()
        }

    raise ValueError(f"unsupported manifest node: {type(node).__name__}")


def _load_wrapper(body: JsonValue, *, result_dir: Path) -> object:
    if not isinstance(body, dict):
        raise ValueError(
            f"malformed Furu wrapper: expected object, got {type(body).__name__}"
        )

    kind = body.get("kind")
    if kind == "external":
        return _load_external(body, result_dir=result_dir)
    if kind == "tuple":
        items = body.get("items")
        if not isinstance(items, list):
            raise ValueError("malformed tuple wrapper: missing items")
        return tuple(_load_value(item, result_dir=result_dir) for item in items)
    if kind == "dataclass":
        return _load_dataclass(body, result_dir=result_dir)
    if kind == "pydantic":
        return _load_pydantic(body, result_dir=result_dir)
    raise ValueError(f"malformed Furu wrapper: unknown kind {kind!r}")


def _load_external(body: dict[str, JsonValue], *, result_dir: Path) -> object:
    codec_id = body.get("codec")
    if not isinstance(codec_id, str):
        raise ValueError("malformed external wrapper: missing codec id")

    rel_path = body.get("path")
    if not isinstance(rel_path, str):
        raise ValueError("malformed external wrapper: missing path")

    artifact_rel = Path(rel_path)
    if artifact_rel.is_absolute():
        raise ValueError(f"external wrapper path must be relative: {rel_path}")

    artifact_dir = (result_dir / artifact_rel).resolve()
    bundle_root = result_dir.resolve()
    artifacts_root = (result_dir / ARTIFACTS_DIR_NAME).resolve()
    try:
        artifact_dir.relative_to(artifacts_root)
        artifact_dir.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError(
            f"external wrapper path escapes bundle artifacts dir: {rel_path}"
        ) from exc

    if not artifact_dir.exists():
        raise ValueError(f"external wrapper artifact directory missing: {artifact_dir}")

    return _codec_by_id(codec_id).load(artifact_dir=artifact_dir, meta=body.get("meta"))


def _load_dataclass(body: dict[str, JsonValue], *, result_dir: Path) -> object:
    type_name = body.get("type")
    if not isinstance(type_name, str):
        raise ValueError("malformed dataclass wrapper: missing type")
    fields_node = body.get("fields")
    if not isinstance(fields_node, dict):
        raise ValueError("malformed dataclass wrapper: missing fields")

    cls = _import_type(type_name)
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
        raise ValueError(
            f"manifest declared dataclass type {type_name!r}, but it is not a dataclass"
        )

    obj = object.__new__(cls)
    for name, child in fields_node.items():
        object.__setattr__(obj, name, _load_value(child, result_dir=result_dir))
    return obj


def _load_pydantic(body: dict[str, JsonValue], *, result_dir: Path) -> object:
    type_name = body.get("type")
    if not isinstance(type_name, str):
        raise ValueError("malformed pydantic wrapper: missing type")
    fields_node = body.get("fields")
    if not isinstance(fields_node, dict):
        raise ValueError("malformed pydantic wrapper: missing fields")

    cls = _import_type(type_name)
    if not (isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)):
        raise ValueError(
            f"manifest declared pydantic type {type_name!r}, but it is not a BaseModel subclass"
        )

    loaded_fields = {
        name: _load_value(child, result_dir=result_dir)
        for name, child in fields_node.items()
    }
    return cls.model_construct(**loaded_fields)


def _import_type(qualified_name: str) -> Any:
    if "." not in qualified_name:
        raise ValueError(f"cannot import unqualified type name: {qualified_name!r}")
    module_name, _, qualname = qualified_name.rpartition(".")
    obj = importlib.import_module(module_name)
    try:
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except AttributeError as exc:
        raise ValueError(
            f"cannot import {qualified_name!r}: attribute not found in module"
        ) from exc
    return obj


def save_result(value: object, result_dir: Path) -> None:
    if result_dir.exists():
        raise FileExistsError(result_dir)

    result_dir.mkdir(parents=True)
    (result_dir / ARTIFACTS_DIR_NAME).mkdir()

    manifest = dump_result_tree(value, result_dir=result_dir)
    _write_manifest(result_dir / MANIFEST_FILE_NAME, manifest)


def load_result(result_dir: Path) -> object:
    manifest_path = result_dir / MANIFEST_FILE_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return load_result_tree(raw, result_dir=result_dir)


def is_complete(result_dir: Path) -> bool:
    return (result_dir / MANIFEST_FILE_NAME).exists()


def _write_manifest(manifest_path: Path, manifest: JsonValue) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True)
    fd = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
