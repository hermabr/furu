from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
from functools import partial
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    assert_never,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic

from furu.result.codec import ResultCodec, ResultRegistry, _default_result_registry
from furu.result.lazy import LazyResult
from furu.result.save_as import _SaveAs
from furu.result.save_as import save_as as save_as
from furu.utils import JsonValue, fully_qualified_name

WRAPPER_KEY: Final = "$furu"
ARTIFACTS_DIR_NAME: Final = "artifacts"
LAZY_DIR_NAME: Final = "lazy"
MANIFEST_FILE_NAME: Final = "manifest.json"
_ROOT_ARTIFACT_NAME: Final = "root"
type ValuePath = tuple[str, ...]
type WrapperKind = Literal[
    "external", "dataclass", "path", "pydantic", "tuple", "set", "frozenset", "lazy"
]


def _child_declared_type(declared_type: object, key: object) -> object:
    declared_type = (
        get_args(declared_type)[0]
        if get_origin(declared_type) is Annotated
        else declared_type
    )
    origin = get_origin(declared_type)
    args = get_args(declared_type)
    if origin is list and args:
        return args[0]
    if origin in (set, frozenset) and args:
        return args[0]
    if origin is dict and len(args) == 2:
        return args[1]
    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        if isinstance(key, int) and key < len(args):
            return args[key]
    return Any


def _value_path_display(value_path: ValuePath) -> str:
    if not value_path:
        return "<root>"
    return "/".join(value_path)


def _validate_result_path_segment(
    value: object,
    *,
    parent_value_path: ValuePath,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"Unsupported result value at {_value_path_display(parent_value_path)}:\n"
            + f"must be strings; got {type(value).__name__} key {value!r}."
        )
    if value == WRAPPER_KEY:
        raise ValueError(
            f"Unsupported result value at {_value_path_display(parent_value_path)}:\n"
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
            f"Unsupported result path at {_value_path_display((*parent_value_path, value))}:\n"
            + "cannot be used as an artifact path segment."
        )
    return value


def _dump_value(
    value: object,
    *,
    declared_type: object = Any,
    value_path: ValuePath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    annotated_codec: type[ResultCodec] | None = None
    if get_origin(declared_type) is Annotated:
        for item in get_args(declared_type)[1:]:
            if isinstance(item, type) and issubclass(item, ResultCodec):
                annotated_codec = item
                break

    match value, annotated_codec:
        case _SaveAs(codec=runtime_codec), annotated_codec if (
            annotated_codec is not None and runtime_codec is not annotated_codec
        ):
            raise TypeError(
                "Conflicting codecs: value was wrapped with furu.save_as(...), "
                "but the field also has an Annotated codec."
            )
        case _SaveAs(value=inner_value, codec=runtime_codec), _:
            return _dump_external(
                inner_value,
                codec=runtime_codec,
                value_path=value_path,
                bundle_dir=bundle_dir,
            )
        case _, annotated_codec if annotated_codec is not None:
            return _dump_external(
                value,
                codec=annotated_codec,
                value_path=value_path,
                bundle_dir=bundle_dir,
            )

    match value:
        case None | bool() | int() | float() | str():
            return value
        case list():
            width = len(str(len(value)))
            return [
                _dump_value(
                    item,
                    declared_type=_child_declared_type(declared_type, i),
                    value_path=(*value_path, f"{i:0{width}d}"),
                    bundle_dir=bundle_dir,
                    registry=registry,
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
                            declared_type=_child_declared_type(declared_type, i),
                            value_path=(*value_path, f"{i:0{width}d}"),
                            bundle_dir=bundle_dir,
                            registry=registry,
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
                            declared_type=_child_declared_type(declared_type, i),
                            value_path=(*value_path, f"{i:0{width}d}"),
                            bundle_dir=bundle_dir,
                            registry=registry,
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
                    parent_value_path=value_path,
                )
                out[key] = _dump_value(
                    child,
                    declared_type=_child_declared_type(declared_type, raw_key),
                    value_path=(*value_path, key),
                    bundle_dir=bundle_dir,
                    registry=registry,
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
            field_types = get_type_hints(value.__class__, include_extras=True)
            for raw_name in value.__class__.model_fields:
                name = _validate_result_path_segment(
                    raw_name, parent_value_path=value_path
                )
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    declared_type=field_types.get(name, Any),
                    value_path=(*value_path, name),
                    bundle_dir=bundle_dir,
                    registry=registry,
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
            field_types = get_type_hints(type(value), include_extras=True)
            for field in dataclasses.fields(cast(Any, value)):
                name = _validate_result_path_segment(
                    field.name, parent_value_path=value_path
                )
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    declared_type=field_types.get(field.name, Any),
                    value_path=(*value_path, name),
                    bundle_dir=bundle_dir,
                    registry=registry,
                )
            return {
                WRAPPER_KEY: {
                    "kind": "dataclass",
                    "type": fully_qualified_name(type(value)),
                    "fields": fields_out,
                }
            }
        case LazyResult():
            lazy_rel = Path(LAZY_DIR_NAME, *(value_path or (_ROOT_ARTIFACT_NAME,)))
            nested_bundle_dir = bundle_dir / lazy_rel
            save_result_bundle(
                value.load(),
                nested_bundle_dir,
                declared_type=(
                    get_args(declared_type)[0]
                    if get_origin(declared_type) is Annotated
                    else declared_type
                ),
                registry=registry,
            )
            return {
                WRAPPER_KEY: {
                    "kind": "lazy",
                    "path": lazy_rel.as_posix(),
                }
            }
        case _:
            if codec := registry.find_codec(value):
                return _dump_external(
                    value,
                    codec=codec,
                    value_path=value_path,
                    bundle_dir=bundle_dir,
                )

    raise ValueError(
        f"Unsupported result value at {_value_path_display(value_path)}:\n"
        f"values of type {type(value).__name__!r} are not supported by Furu. Add a custom codec"
    )


def _dump_external(
    value: object,
    *,
    codec: type[ResultCodec],
    value_path: ValuePath,
    bundle_dir: Path,
) -> JsonValue:
    artifact_rel = Path(ARTIFACTS_DIR_NAME, *(value_path or (_ROOT_ARTIFACT_NAME,)))
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


def _import_type(qualified_name: str) -> Any:
    module_name, _, attr_name = qualified_name.rpartition(".")
    return getattr(importlib.import_module(module_name), attr_name)


def _load_value(
    node: JsonValue,
    *,
    bundle_dir: Path,
    value_path: ValuePath,
    registry: ResultRegistry,
) -> object:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case list():
            width = len(str(len(node)))
            return [
                _load_value(
                    child,
                    bundle_dir=bundle_dir,
                    value_path=(*value_path, f"{i:0{width}d}"),
                    registry=registry,
                )
                for i, child in enumerate(node)
            ]
        case dict() if WRAPPER_KEY in node:
            return _load_wrapper(
                cast(dict[str, Any], node[WRAPPER_KEY]),
                bundle_dir=bundle_dir,
                value_path=value_path,
                registry=registry,
            )
        case dict():
            return {
                key: _load_value(
                    child,
                    bundle_dir=bundle_dir,
                    value_path=(*value_path, key),
                    registry=registry,
                )
                for key, child in node.items()
            }
        case _:
            assert_never(node)


def _load_validated_fields(
    *,
    kind: str,
    cls: type[Any],
    expected: set[str],
    raw_fields: dict[str, JsonValue],
    bundle_dir: Path,
    value_path: ValuePath,
    registry: ResultRegistry,
) -> dict[str, object]:
    actual = set(raw_fields)
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        return {
            name: _load_value(
                child,
                bundle_dir=bundle_dir,
                value_path=(*value_path, name),
                registry=registry,
            )
            for name, child in raw_fields.items()
        }

    details: list[str] = []
    if missing:
        details.append("missing fields: " + ", ".join(sorted(missing)))
    if extra:
        details.append("extra fields: " + ", ".join(sorted(extra)))

    raise ValueError(
        f"Cannot load {kind} {fully_qualified_name(cls)} at {_value_path_display(value_path)}: "
        + "; ".join(details)
    )


def _load_wrapper(
    body: dict[str, Any],
    *,
    bundle_dir: Path,
    value_path: ValuePath,
    registry: ResultRegistry,
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

            return registry.resolve_codec(body["codec"]).load(artifact_dir=artifact_dir)
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
                    registry=registry,
                )
            )
        case "dataclass":
            cls = _import_type(body["type"])
            if not dataclasses.is_dataclass(cls):
                raise ValueError(
                    f"Cannot load dataclass at {_value_path_display(value_path)}: "
                    f"{fully_qualified_name(cls)} is not a dataclass"
                )
            dataclass_fields = dataclasses.fields(cls)
            init_fields = {field.name for field in dataclass_fields if field.init}
            loaded_fields = _load_validated_fields(
                kind="dataclass",
                cls=cls,
                expected={field.name for field in dataclass_fields},
                raw_fields=body["fields"],
                bundle_dir=bundle_dir,
                value_path=value_path,
                registry=registry,
            )
            try:
                return cls(
                    **{
                        name: value
                        for name, value in loaded_fields.items()
                        if name in init_fields
                    }
                )
            except Exception as exc:
                raise ValueError(
                    f"Cannot load dataclass {fully_qualified_name(cls)} "
                    f"at {_value_path_display(value_path)}: {exc}"
                ) from exc
        case "path":
            return Path(body["value"])
        case "tuple":
            return tuple(
                _load_value(
                    child,
                    bundle_dir=bundle_dir,
                    value_path=(*value_path, str(i)),
                    registry=registry,
                )
                for i, child in enumerate(body["items"])
            )
        case "set":
            return {
                _load_value(
                    child,
                    bundle_dir=bundle_dir,
                    value_path=(*value_path, str(i)),
                    registry=registry,
                )
                for i, child in enumerate(body["items"])
            }
        case "frozenset":
            return frozenset(
                _load_value(
                    child,
                    bundle_dir=bundle_dir,
                    value_path=(*value_path, str(i)),
                    registry=registry,
                )
                for i, child in enumerate(body["items"])
            )
        case "pydantic":
            cls = _import_type(body["type"])
            if not issubclass(cls, pydantic.BaseModel):
                raise ValueError(
                    f"Cannot load pydantic model at {_value_path_display(value_path)}: "
                    f"{fully_qualified_name(cls)} is not a pydantic model"
                )
            loaded_fields = _load_validated_fields(
                kind="pydantic model",
                cls=cls,
                expected=set(cls.model_fields),
                raw_fields=body["fields"],
                bundle_dir=bundle_dir,
                value_path=value_path,
                registry=registry,
            )
            try:
                return cls.model_validate(loaded_fields)
            except pydantic.ValidationError as exc:
                raise ValueError(
                    f"Cannot load pydantic model {fully_qualified_name(cls)} "
                    f"at {_value_path_display(value_path)}: {exc}"
                ) from exc
        case _:
            raise ValueError(f"unknown wrapper kind: {kind!r}")


def save_result_bundle(
    value: object,
    bundle_dir: Path,
    *,
    declared_type: object = Any,
    registry: ResultRegistry | None = None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=False)
    registry = registry or _default_result_registry()

    manifest = _dump_value(
        value,
        declared_type=declared_type,
        value_path=(),
        bundle_dir=bundle_dir,
        registry=registry,
    )
    (bundle_dir / MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def load_result_bundle(
    bundle_dir: Path,
    *,
    registry: ResultRegistry | None = None,
) -> object:
    registry = registry or _default_result_registry()
    manifest_path = bundle_dir / MANIFEST_FILE_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _load_value(raw, bundle_dir=bundle_dir, value_path=(), registry=registry)
