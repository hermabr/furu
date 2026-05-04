from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import os
from abc import ABC, abstractmethod
from functools import cache
from pathlib import Path
from typing import Any, Final, Literal, assert_never, cast

import pydantic

from furu.utils import JsonValue, fully_qualified_name


WRAPPER_KEY: Final[str] = "$furu"
ARTIFACTS_DIR_NAME: Final[str] = "artifacts"
MANIFEST_FILE_NAME: Final[str] = "manifest.json"
_ROOT_ARTIFACT_NAME: Final[str] = "root"
type LogicalPath = tuple[str, ...]
type WrapperKind = Literal["external", "dataclass", "pydantic"]


def _path_display(path: LogicalPath) -> str:
    if not path:
        return "<root>"
    return "/".join(path)


def _is_safe_path_segment(text: str) -> bool:
    if text == "" or text == "." or text == "..":
        return False
    if "/" in text or "\\" in text:
        return False
    if "\x00" in text:
        return False
    return True


class ResultCodec(ABC):
    @classmethod
    def codec_id(cls) -> str:
        return fully_qualified_name(cls)

    @classmethod
    def dependencies_available(cls) -> bool:
        return True

    @classmethod
    @abstractmethod
    def matches(cls, value: object) -> bool:
        pass

    @classmethod
    @abstractmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        pass

    @classmethod
    @abstractmethod
    def load(cls, *, artifact_dir: Path, meta: JsonValue) -> object:
        pass


class NumpyNpyCodec(ResultCodec):
    @classmethod
    def dependencies_available(cls) -> bool:
        return importlib.util.find_spec("numpy") is not None

    @classmethod
    def matches(cls, value: object) -> bool:
        import numpy as np

        return isinstance(value, np.ndarray)

    @classmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        import numpy as np

        array = cast("np.ndarray[Any, Any]", value)
        if array.dtype.hasobject:
            raise ValueError(
                f"Unsupported result value at {_path_display(path)}:\n"
                "numpy object-dtype arrays are not supported by the default npy codec."
            )

        np.save(artifact_dir / "data.npy", array, allow_pickle=False)

        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    @classmethod
    def load(cls, *, artifact_dir: Path, meta: JsonValue) -> object:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


@cache
def default_codecs() -> dict[str, type[ResultCodec]]:
    codecs: dict[str, type[ResultCodec]] = {}
    for codec in (NumpyNpyCodec,):
        codec_id = codec.codec_id()
        if codec_id in codecs:
            raise ValueError(f"duplicate result codec id: {codec_id}")
        if not codec.dependencies_available():
            continue
        codecs[codec_id] = codec
    return codecs


def _dump_value(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    codecs: dict[str, type[ResultCodec]],
) -> JsonValue:
    match value:
        case None | bool() | int() | float() | str():
            return value
        case list():
            width = max(len(str(len(value))), 1)
            return [
                _dump_value(
                    item,
                    path=(*path, f"{i:0{width}d}"),
                    bundle_dir=bundle_dir,
                    codecs=codecs,
                )
                for i, item in enumerate(value)
            ]
        case dict():
            out: dict[str, JsonValue] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"Unsupported result value at {_path_display(path)}:\n"
                        f"dict result keys must be strings in Stage 1; got {type(key).__name__} key {key!r}."
                    )
                if key == WRAPPER_KEY:
                    raise ValueError(
                        f"Unsupported result value at {_path_display(path)}:\n"
                        f"dict keys named {WRAPPER_KEY!r} are reserved by Furu result persistence."
                    )
                if not _is_safe_path_segment(key):
                    raise ValueError(
                        f"Unsupported result path at {_path_display((*path, key))}:\n"
                        "dict key cannot be used as an artifact path segment."
                    )
                out[key] = _dump_value(
                    child,
                    path=(*path, key),
                    bundle_dir=bundle_dir,
                    codecs=codecs,
                )
            return out
        case pydantic.BaseModel():
            fields_out: dict[str, JsonValue] = {}
            for name in value.__class__.model_fields:
                if not _is_safe_path_segment(name):
                    raise ValueError(
                        f"Unsupported result path at {_path_display((*path, name))}:\n"
                        "pydantic field name cannot be used as an artifact path segment."
                    )
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    path=(*path, name),
                    bundle_dir=bundle_dir,
                    codecs=codecs,
                )
            return {
                WRAPPER_KEY: {
                    "kind": "pydantic",
                    "type": fully_qualified_name(type(value)),
                    "fields": fields_out,
                }
            }
        case _ if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields_out: dict[str, JsonValue] = {}
            for field in dataclasses.fields(cast(Any, value)):
                if not _is_safe_path_segment(field.name):
                    raise ValueError(
                        f"Unsupported result path at {_path_display((*path, field.name))}:\n"
                        "dataclass field name cannot be used as an artifact path segment."
                    )
                fields_out[field.name] = _dump_value(
                    getattr(value, field.name),
                    path=(*path, field.name),
                    bundle_dir=bundle_dir,
                    codecs=codecs,
                )
            return {
                WRAPPER_KEY: {
                    "kind": "dataclass",
                    "type": fully_qualified_name(type(value)),
                    "fields": fields_out,
                }
            }
        case _:
            for codec in codecs.values():
                if codec.matches(value):
                    artifact_rel = (
                        Path(ARTIFACTS_DIR_NAME) / _ROOT_ARTIFACT_NAME
                        if not path
                        else Path(ARTIFACTS_DIR_NAME, *path)
                    )
                    artifact_dir = bundle_dir / artifact_rel
                    artifact_dir.mkdir(parents=True, exist_ok=False)
                    meta = codec.dump(value, artifact_dir=artifact_dir, path=path)
                    return {
                        WRAPPER_KEY: {
                            "kind": "external",
                            "codec": codec.codec_id(),
                            "path": artifact_rel.as_posix(),
                            "meta": meta,
                        }
                    }

            raise ValueError(
                f"Unsupported result value at {_path_display(path)}:\n"
                f"values of type {type(value).__name__!r} are not supported by Furu Stage 1."
            )


def _load_value(
    node: JsonValue,
    *,
    bundle_dir: Path,
    codecs: dict[str, type[ResultCodec]],
) -> object:
    if node is None or isinstance(node, (bool, int, float, str)):
        return node

    if isinstance(node, list):
        return [
            _load_value(child, bundle_dir=bundle_dir, codecs=codecs) for child in node
        ]

    if isinstance(node, dict):
        if WRAPPER_KEY in node:
            return _load_wrapper(
                cast(dict[str, Any], node[WRAPPER_KEY]),
                bundle_dir=bundle_dir,
                codecs=codecs,
            )
        return {
            key: _load_value(child, bundle_dir=bundle_dir, codecs=codecs)
            for key, child in node.items()
        }

    raise ValueError(f"unsupported manifest node: {type(node).__name__}")


def _load_wrapper(
    body: dict[str, Any],
    *,
    bundle_dir: Path,
    codecs: dict[str, type[ResultCodec]],
) -> object:
    kind = cast(WrapperKind, body["kind"])
    match kind:
        case "external":
            codec_id: str = body["codec"]
            rel_path: str = body["path"]
            artifact_rel = Path(rel_path)
            if artifact_rel.is_absolute():
                raise ValueError(f"external wrapper path must be relative: {rel_path}")

            artifact_dir = (bundle_dir / artifact_rel).resolve()
            artifacts_root = (bundle_dir / ARTIFACTS_DIR_NAME).resolve()
            try:
                artifact_dir.relative_to(artifacts_root)
            except ValueError as exc:
                raise ValueError(
                    f"external wrapper path escapes bundle artifacts dir: {rel_path}"
                ) from exc

            if not artifact_dir.exists():
                raise ValueError(
                    f"external wrapper artifact directory missing: {artifact_dir}"
                )

            if codec_id not in codecs:
                raise ValueError(f"unknown result codec: {codec_id}")
            return codecs[codec_id].load(artifact_dir=artifact_dir, meta=body["meta"])
        case "dataclass":
            cls = _import_type(body["type"])
            loaded_fields = {
                name: _load_value(child, bundle_dir=bundle_dir, codecs=codecs)
                for name, child in body["fields"].items()
            }

            obj = object.__new__(cls)
            for name, value in loaded_fields.items():
                object.__setattr__(obj, name, value)
            return obj
        case "pydantic":
            cls = _import_type(body["type"])
            loaded_fields = {
                name: _load_value(child, bundle_dir=bundle_dir, codecs=codecs)
                for name, child in body["fields"].items()
            }
            return cls.model_construct(**loaded_fields)
        case _:
            assert_never(kind)


def _import_type(qualified_name: str) -> Any:
    module_name, _, attr_name = qualified_name.rpartition(".")
    return getattr(importlib.import_module(module_name), attr_name)


def save_result_bundle(value: object, bundle_dir: Path) -> None:
    if bundle_dir.exists():
        raise FileExistsError(bundle_dir)

    bundle_dir.mkdir(parents=True)
    (bundle_dir / ARTIFACTS_DIR_NAME).mkdir()

    codecs = default_codecs()

    manifest = _dump_value(
        value,
        path=(),
        bundle_dir=bundle_dir,
        codecs=codecs,
    )
    _write_manifest(bundle_dir / MANIFEST_FILE_NAME, manifest)


def load_result_bundle[T](bundle_dir: Path) -> T:
    manifest_path = bundle_dir / MANIFEST_FILE_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    codecs = default_codecs()
    return cast(T, _load_value(raw, bundle_dir=bundle_dir, codecs=codecs))


def result_bundle_is_complete(bundle_dir: Path) -> bool:
    return (bundle_dir / MANIFEST_FILE_NAME).exists()


def _write_manifest(manifest_path: Path, manifest: JsonValue) -> None:
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
