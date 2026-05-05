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


def _validate_result_path_segment(
    value: object,
    *,
    parent_path: LogicalPath,
    source: Literal["dataclass field name", "dict key", "pydantic field name"],
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"Unsupported result value at {_path_display(parent_path)}:\n"
            f"dict result keys must be strings in Stage 1; got {type(value).__name__} key {value!r}."
        )
    if value == WRAPPER_KEY:
        subject = "dict keys" if source == "dict key" else f"{source}s"
        raise ValueError(
            f"Unsupported result value at {_path_display(parent_path)}:\n"
            f"{subject} named {WRAPPER_KEY!r} are reserved by Furu result persistence."
        )
    if not _is_safe_path_segment(value):
        raise ValueError(
            f"Unsupported result path at {_path_display((*parent_path, value))}:\n"
            f"{source} cannot be used as an artifact path segment."
        )
    return value


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
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, *, artifact_dir: Path) -> object:
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
    ) -> None:
        import numpy as np

        if not isinstance(value, np.ndarray):
            raise ValueError
        np.save(artifact_dir / "data.npy", value, allow_pickle=False)

    @classmethod
    def load(cls, *, artifact_dir: Path) -> object:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


@cache
def default_codecs() -> dict[str, type[ResultCodec]]:
    codecs: dict[str, type[ResultCodec]] = {}
    for codec in (NumpyNpyCodec,):
        if not codec.dependencies_available():
            continue
        codecs[codec.codec_id()] = codec
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
            for raw_key, child in value.items():
                key = _validate_result_path_segment(
                    raw_key,
                    parent_path=path,
                    source="dict key",
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
            for raw_name in value.__class__.model_fields:
                name = _validate_result_path_segment(
                    raw_name,
                    parent_path=path,
                    source="pydantic field name",
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
                name = _validate_result_path_segment(
                    field.name,
                    parent_path=path,
                    source="dataclass field name",
                )
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    path=(*path, name),
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
                    codec.dump(value, artifact_dir=artifact_dir)
                    return {
                        WRAPPER_KEY: {
                            "kind": "external",
                            "codec": codec.codec_id(),
                            "path": artifact_rel.as_posix(),
                        }
                    }

    raise ValueError(
        f"Unsupported result value at {_path_display(path)}:\n"
        f"values of type {type(value).__name__!r} are not supported by Furu. Add a custom codec"
    )


def _load_value(
    node: JsonValue,
    *,
    bundle_dir: Path,
    codecs: dict[str, type[ResultCodec]],
) -> object:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case list():
            return [
                _load_value(child, bundle_dir=bundle_dir, codecs=codecs)
                for child in node
            ]
        case dict() if WRAPPER_KEY in node:
            return _load_wrapper(
                cast(dict[str, Any], node[WRAPPER_KEY]),
                bundle_dir=bundle_dir,
                codecs=codecs,
            )
        case dict():
            return {
                key: _load_value(child, bundle_dir=bundle_dir, codecs=codecs)
                for key, child in node.items()
            }
        case _:
            assert_never(node)


def _import_type(qualified_name: str) -> Any:
    module_name, _, attr_name = qualified_name.rpartition(".")
    return getattr(importlib.import_module(module_name), attr_name)


def _load_wrapper(
    body: dict[str, Any],
    *,
    bundle_dir: Path,
    codecs: dict[str, type[ResultCodec]],
) -> object:
    kind: WrapperKind = body["kind"]
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

            return codecs[codec_id].load(artifact_dir=artifact_dir)
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


def save_result_bundle(value: object, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=False)
    (bundle_dir / ARTIFACTS_DIR_NAME).mkdir()

    codecs = default_codecs()

    manifest = _dump_value(
        value,
        path=(),
        bundle_dir=bundle_dir,
        codecs=codecs,
    )
    with (bundle_dir / MANIFEST_FILE_NAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def load_result_bundle[T](bundle_dir: Path) -> T:
    manifest_path = bundle_dir / MANIFEST_FILE_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    codecs = default_codecs()
    return cast(T, _load_value(raw, bundle_dir=bundle_dir, codecs=codecs))
