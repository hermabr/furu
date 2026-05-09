from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal, assert_never, cast

import pydantic

from furu.result.codec import _DEFAULT_CODECS
from furu.result.lazy import LazyResult
from furu.utils import JsonValue, fully_qualified_name

WRAPPER_KEY: Final = "$furu"
ARTIFACTS_DIR_NAME: Final = "artifacts"
LAZY_DIR_NAME: Final = "lazy"
MANIFEST_FILE_NAME: Final = "manifest.json"
_ROOT_ARTIFACT_NAME: Final = "root"
type LogicalPath = tuple[str, ...]
type WrapperKind = Literal[
    "external", "dataclass", "path", "pydantic", "tuple", "set", "frozenset", "lazy"
]


def _path_display(path: LogicalPath) -> str:
    if not path:
        return "<root>"
    return "/".join(path)


def _validate_result_path_segment(
    value: object,
    *,
    parent_path: LogicalPath,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"Unsupported result value at {_path_display(parent_path)}:\n"
            + f"must be strings; got {type(value).__name__} key {value!r}."
        )
    if value == WRAPPER_KEY:
        raise ValueError(
            f"Unsupported result value at {_path_display(parent_path)}:\n"
            + f"named {WRAPPER_KEY!r} are reserved by Furu result persistence."
        )
    if (
        value == ""
        or value == "."
        or value == ".."
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(
            f"Unsupported result path at {_path_display((*parent_path, value))}:\n"
            + "cannot be used as an artifact path segment."
        )
    return value


def _dump_value(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
) -> JsonValue:
    match value:
        case None | bool() | int() | float() | str():
            return value
        case list():
            width = len(str(len(value)))
            return [
                _dump_value(
                    item,
                    path=(*path, f"{i:0{width}d}"),
                    bundle_dir=bundle_dir,
                )
                for i, item in enumerate(value)
            ]
        case tuple():
            width = len(str(len(value)))
            return {
                WRAPPER_KEY: {
                    "kind": "tuple",
                    "items": [
                        _dump_value(
                            item,
                            path=(*path, f"{i:0{width}d}"),
                            bundle_dir=bundle_dir,
                        )
                        for i, item in enumerate(value)
                    ],
                }
            }
        case set() | frozenset():
            kind = "frozenset" if isinstance(value, frozenset) else "set"
            items = sorted(
                value,
                key=lambda item: (
                    type(item).__module__,
                    type(item).__qualname__,
                    repr(item),
                ),
            )
            width = len(str(len(items)))
            return {
                WRAPPER_KEY: {
                    "kind": kind,
                    "items": [
                        _dump_value(
                            item,
                            path=(*path, f"{i:0{width}d}"),
                            bundle_dir=bundle_dir,
                        )
                        for i, item in enumerate(items)
                    ],
                }
            }
        case dict():
            out: dict[str, JsonValue] = {}
            for raw_key, child in value.items():
                key = _validate_result_path_segment(
                    raw_key,
                    parent_path=path,
                )
                out[key] = _dump_value(
                    child,
                    path=(*path, key),
                    bundle_dir=bundle_dir,
                )
            return out
        case Path():
            return {
                WRAPPER_KEY: {
                    "kind": "path",
                    "value": str(value),
                }
            }
        case pydantic.BaseModel():
            fields_out: dict[str, JsonValue] = {}
            for raw_name in value.__class__.model_fields:
                name = _validate_result_path_segment(raw_name, parent_path=path)
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    path=(*path, name),
                    bundle_dir=bundle_dir,
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
                name = _validate_result_path_segment(field.name, parent_path=path)
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    path=(*path, name),
                    bundle_dir=bundle_dir,
                )
            return {
                WRAPPER_KEY: {
                    "kind": "dataclass",
                    "type": fully_qualified_name(type(value)),
                    "fields": fields_out,
                }
            }
        case LazyResult():
            lazy_rel = Path(LAZY_DIR_NAME, *(path or (_ROOT_ARTIFACT_NAME,)))
            nested_bundle_dir = bundle_dir / lazy_rel
            save_result_bundle(value.load(), nested_bundle_dir)
            return {
                WRAPPER_KEY: {
                    "kind": "lazy",
                    "path": lazy_rel.as_posix(),
                }
            }
        case _:
            for codec in _DEFAULT_CODECS.values():
                if codec.matches(value):
                    artifact_rel = Path(
                        ARTIFACTS_DIR_NAME, *(path or (_ROOT_ARTIFACT_NAME,))
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


def _import_type(qualified_name: str) -> Any:
    module_name, _, attr_name = qualified_name.rpartition(".")
    return getattr(importlib.import_module(module_name), attr_name)


def _load_value(
    node: JsonValue,
    *,
    bundle_dir: Path,
) -> object:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case list():
            return [_load_value(child, bundle_dir=bundle_dir) for child in node]
        case dict() if WRAPPER_KEY in node:
            return _load_wrapper(
                cast(dict[str, Any], node[WRAPPER_KEY]),
                bundle_dir=bundle_dir,
            )
        case dict():
            return {
                key: _load_value(child, bundle_dir=bundle_dir)
                for key, child in node.items()
            }
        case _:
            assert_never(node)


def _load_wrapper(
    body: dict[str, Any],
    *,
    bundle_dir: Path,
) -> object:
    kind: WrapperKind = body["kind"]
    match kind:
        case "external":
            artifact_rel = Path(body["path"])
            if artifact_rel.is_absolute():
                raise ValueError(
                    f"external wrapper path must be relative: {artifact_rel}"
                )

            artifact_dir = (bundle_dir / artifact_rel).resolve()
            artifacts_root = (bundle_dir / ARTIFACTS_DIR_NAME).resolve()
            if not artifact_dir.is_relative_to(artifacts_root):
                raise ValueError(
                    f"external wrapper path escapes bundle artifacts dir: {artifact_rel}"
                )

            if not artifact_dir.exists():
                raise ValueError(
                    f"external wrapper artifact directory missing: {artifact_dir}"
                )

            return _DEFAULT_CODECS[body["codec"]].load(artifact_dir=artifact_dir)
        case "lazy":
            if (nested_rel := Path(body["path"])).is_absolute():
                raise ValueError(f"lazy wrapper path must be relative: {nested_rel}")

            nested_bundle_dir = (bundle_dir / nested_rel).resolve()
            lazy_root = (bundle_dir / LAZY_DIR_NAME).resolve()
            if not nested_bundle_dir.is_relative_to(lazy_root):
                raise ValueError(
                    f"lazy wrapper path escapes bundle lazy dir: {nested_rel}"
                )

            if not (nested_bundle_dir / MANIFEST_FILE_NAME).exists():
                raise ValueError(
                    f"lazy wrapper nested manifest missing: {nested_bundle_dir}"
                )

            return LazyResult._from_loader(
                partial(
                    load_result_bundle,
                    bundle_dir=nested_bundle_dir,
                )
            )
        case "dataclass":  # TODO: do validation on the dataclass/pydantic object, so that we know the new object has exactly the same fields as the old one
            cls = _import_type(body["type"])
            loaded_fields = {
                name: _load_value(child, bundle_dir=bundle_dir)
                for name, child in body["fields"].items()
            }
            obj = object.__new__(cls)
            for name, value in loaded_fields.items():
                object.__setattr__(obj, name, value)
            return obj
        case "path":
            return Path(body["value"])
        case "tuple":
            return tuple(
                _load_value(child, bundle_dir=bundle_dir) for child in body["items"]
            )
        case "set":
            return {
                _load_value(child, bundle_dir=bundle_dir) for child in body["items"]
            }
        case "frozenset":
            return frozenset(
                _load_value(child, bundle_dir=bundle_dir) for child in body["items"]
            )
        case "pydantic":
            cls = _import_type(body["type"])
            loaded_fields = {
                name: _load_value(child, bundle_dir=bundle_dir)
                for name, child in body["fields"].items()
            }
            return cls.model_construct(**loaded_fields)
        case _:
            raise ValueError(f"unknown wrapper kind: {kind!r}")


def save_result_bundle(value: object, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=False)

    manifest = _dump_value(
        value,
        path=(),
        bundle_dir=bundle_dir,
    )
    (bundle_dir / MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def load_result_bundle(bundle_dir: Path) -> object:
    manifest_path = bundle_dir / MANIFEST_FILE_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _load_value(raw, bundle_dir=bundle_dir)
