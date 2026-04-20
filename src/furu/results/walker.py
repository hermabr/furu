from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from furu.utils import JsonValue, _stable_json_dump, fully_qualified_name

from .config import ResultConfig
from .errors import ResultCodecError, ResultLoadError, ResultSerializationError
from .lazy import FuruLazy
from .markers import ResultSpec, SaveWith, merge_specs, unwrap_furu_result
from .paths import LogicalPath, artifact_relpath_for
from .protocol import DumpContext, LoadContext, ResultCodec

_WRAPPER_KEY = "$furu"


@dataclass(slots=True)
class DumpNodeContext:
    bundle_dir: Path
    config: ResultConfig
    logical_path: LogicalPath = LogicalPath()
    active_ids: set[int] = field(default_factory=set)

    @property
    def registry(self):
        return self.config.registry

    def child(self, token: str | int) -> "DumpNodeContext":
        return DumpNodeContext(
            bundle_dir=self.bundle_dir,
            config=self.config,
            logical_path=self.logical_path.child(token),
            active_ids=self.active_ids,
        )


def dump_root(value: Any, *, bundle_dir: Path, config: ResultConfig) -> JsonValue:
    return dump_node(value, DumpNodeContext(bundle_dir=bundle_dir, config=config))


def load_root(root: JsonValue, *, bundle_dir: Path, config: ResultConfig) -> Any:
    return load_node(
        root,
        bundle_dir=bundle_dir,
        config=config,
        logical_path=LogicalPath(),
    )


def dump_node(
    value: Any,
    ctx: DumpNodeContext,
    *,
    field_spec: ResultSpec | None = None,
) -> JsonValue:
    wrapper_spec, value = unwrap_furu_result(value)
    path_spec = ctx.config.rule_for_path(ctx.logical_path)
    spec = merge_specs(field_spec, path_spec, wrapper_spec)

    if spec is not None and spec.codec is not None:
        codec = _resolve_codec_ref(ctx.config, spec.codec)
        return _dump_external(value, ctx, codec=codec, lazy=bool(spec.lazy))

    if _supports_custom_protocol(value):
        return _dump_custom_protocol(
            value, ctx, lazy=bool(spec.lazy) if spec else False
        )

    codec = ctx.registry.resolve_for_value(value)
    if codec is not None:
        return _dump_external(
            value, ctx, codec=codec, lazy=bool(spec.lazy) if spec else False
        )

    if spec is not None and spec.lazy is True:
        if _is_json_compatible(value, allow_reserved_key=True):
            json_codec = ctx.registry.get_codec("furu.json.v1")
            return _dump_external(value, ctx, codec=json_codec, lazy=True)
        raise ResultSerializationError(
            ctx.logical_path,
            type(value),
            "lazy=True requires an external codec; use furu.lazy(value, codec=...) or register a codec.",
        )

    match value:
        case None | bool() | int() | str():
            return value
        case float():
            if not math.isfinite(value):
                raise ResultSerializationError(
                    ctx.logical_path,
                    type(value),
                    "non-finite floats are not supported in result manifests.",
                )
            return value
        case list():
            with _cycle_guard(value, ctx):
                return [dump_node(item, ctx.child(i)) for i, item in enumerate(value)]
        case dict():
            with _cycle_guard(value, ctx):
                return _dump_mapping(value, ctx)
        case tuple():
            with _cycle_guard(value, ctx):
                return _tagged(
                    "tuple",
                    items=[
                        dump_node(item, ctx.child(i)) for i, item in enumerate(value)
                    ],
                )
        case set():
            with _cycle_guard(value, ctx):
                return _tagged("set", items=_dump_sorted_set_items(value, ctx))
        case frozenset():
            with _cycle_guard(value, ctx):
                return _tagged("frozenset", items=_dump_sorted_set_items(value, ctx))
        case _ if _is_dataclass_instance(value):
            with _cycle_guard(value, ctx):
                return _dump_dataclass(value, ctx)
        case BaseModel():
            with _cycle_guard(value, ctx):
                return _dump_pydantic(value, ctx)
        case _:
            raise ResultSerializationError(ctx.logical_path, type(value))


def load_node(
    node: JsonValue,
    *,
    bundle_dir: Path,
    config: ResultConfig,
    logical_path: LogicalPath,
) -> Any:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case list():
            return [
                load_node(
                    item,
                    bundle_dir=bundle_dir,
                    config=config,
                    logical_path=logical_path.child(i),
                )
                for i, item in enumerate(node)
            ]
        case dict() if _WRAPPER_KEY in node:
            if set(node) != {_WRAPPER_KEY}:
                raise ResultLoadError(
                    logical_path,
                    f"wrapper nodes may only contain the reserved key {_WRAPPER_KEY!r}.",
                )
            raw_wrapper = node[_WRAPPER_KEY]
            if not isinstance(raw_wrapper, dict):
                raise ResultLoadError(
                    logical_path, "wrapper payload must be a JSON object."
                )
            kind = raw_wrapper.get("kind")
            if not isinstance(kind, str):
                raise ResultLoadError(
                    logical_path, "wrapper payload is missing a string kind."
                )
            return _load_wrapped_node(
                kind,
                raw_wrapper,
                bundle_dir=bundle_dir,
                config=config,
                logical_path=logical_path,
            )
        case dict():
            return {
                key: load_node(
                    value,
                    bundle_dir=bundle_dir,
                    config=config,
                    logical_path=logical_path.child(key),
                )
                for key, value in node.items()
            }
        case _:
            raise ResultLoadError(logical_path, "unsupported manifest value.")


def _load_wrapped_node(
    kind: str,
    node: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    config: ResultConfig,
    logical_path: LogicalPath,
) -> Any:
    match kind:
        case "external":
            return _load_external_node(
                node,
                bundle_dir=bundle_dir,
                config=config,
                logical_path=logical_path,
            )
        case "custom":
            return _load_custom_node(
                node,
                bundle_dir=bundle_dir,
                config=config,
                logical_path=logical_path,
            )
        case "dataclass":
            python_type = _require_str(node, "python_type", logical_path)
            fields = _require_dict(node, "fields", logical_path)
            cls = _import_type(python_type, logical_path)
            loaded_fields = {
                key: load_node(
                    value,
                    bundle_dir=bundle_dir,
                    config=config,
                    logical_path=logical_path.child(key),
                )
                for key, value in fields.items()
            }
            return cls(**loaded_fields)
        case "pydantic":
            python_type = _require_str(node, "python_type", logical_path)
            fields = _require_dict(node, "fields", logical_path)
            cls = _import_type(python_type, logical_path)
            loaded_fields = {
                key: load_node(
                    value,
                    bundle_dir=bundle_dir,
                    config=config,
                    logical_path=logical_path.child(key),
                )
                for key, value in fields.items()
            }
            return cls.model_construct(**loaded_fields)
        case "tuple":
            items = _require_list(node, "items", logical_path)
            return tuple(
                load_node(
                    item,
                    bundle_dir=bundle_dir,
                    config=config,
                    logical_path=logical_path.child(i),
                )
                for i, item in enumerate(items)
            )
        case "set":
            items = _require_list(node, "items", logical_path)
            return {
                load_node(
                    item,
                    bundle_dir=bundle_dir,
                    config=config,
                    logical_path=logical_path.child(i),
                )
                for i, item in enumerate(items)
            }
        case "frozenset":
            items = _require_list(node, "items", logical_path)
            return frozenset(
                load_node(
                    item,
                    bundle_dir=bundle_dir,
                    config=config,
                    logical_path=logical_path.child(i),
                )
                for i, item in enumerate(items)
            )
        case "mapping":
            items = _require_list(node, "items", logical_path)
            loaded: dict[str, Any] = {}
            for item in items:
                if not isinstance(item, dict):
                    raise ResultLoadError(
                        logical_path, "mapping items must be JSON objects."
                    )
                key = item.get("key")
                if not isinstance(key, str):
                    raise ResultLoadError(
                        logical_path, "mapping item keys must be strings."
                    )
                loaded[key] = load_node(
                    item.get("value"),
                    bundle_dir=bundle_dir,
                    config=config,
                    logical_path=logical_path.child(key),
                )
            return loaded
        case _:
            raise ResultLoadError(logical_path, f"unknown wrapper kind {kind!r}.")


def _dump_external(
    value: Any,
    ctx: DumpNodeContext,
    *,
    codec: ResultCodec[Any],
    lazy: bool,
) -> JsonValue:
    artifact_relpath = artifact_relpath_for(ctx.logical_path)
    artifact_dir = ctx.bundle_dir / artifact_relpath
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump_ctx = DumpContext(
        bundle_dir=ctx.bundle_dir,
        artifact_dir=artifact_dir,
        artifact_relpath=artifact_relpath,
        logical_path=ctx.logical_path,
        registry=ctx.registry,
    )
    try:
        meta = codec.dump(value, dump_ctx)
    except Exception as exc:
        raise ResultSerializationError(
            ctx.logical_path,
            type(value),
            f"codec {codec.codec_id!r} failed while saving this value: {exc}",
        ) from exc

    if meta is not None and not _is_json_compatible(meta, allow_reserved_key=True):
        raise ResultSerializationError(
            ctx.logical_path,
            type(value),
            f"codec {codec.codec_id!r} returned non-JSON metadata.",
        )

    wrapped: dict[str, JsonValue] = {
        "kind": "external",
        "codec": codec.codec_id,
        "artifact_dir": artifact_relpath,
        "lazy": lazy,
    }
    python_type = _maybe_python_type(type(value))
    if python_type is not None:
        wrapped["python_type"] = python_type
    if meta is not None:
        wrapped["meta"] = meta
    return _tagged_dict(wrapped)


def _dump_custom_protocol(value: Any, ctx: DumpNodeContext, *, lazy: bool) -> JsonValue:
    artifact_relpath = artifact_relpath_for(ctx.logical_path)
    artifact_dir = ctx.bundle_dir / artifact_relpath
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump_ctx = DumpContext(
        bundle_dir=ctx.bundle_dir,
        artifact_dir=artifact_dir,
        artifact_relpath=artifact_relpath,
        logical_path=ctx.logical_path,
        registry=ctx.registry,
    )
    try:
        payload = value.__furu_result_dump__(dump_ctx)
    except Exception as exc:
        raise ResultSerializationError(
            ctx.logical_path,
            type(value),
            f"custom result protocol failed while saving this value: {exc}",
        ) from exc

    if not _is_json_compatible(payload, allow_reserved_key=True):
        raise ResultSerializationError(
            ctx.logical_path,
            type(value),
            "custom result payload must be JSON-compatible.",
        )

    python_type = _maybe_python_type(type(value))
    if python_type is None:
        raise ResultSerializationError(
            ctx.logical_path,
            type(value),
            "custom result protocol types must be importable by fully qualified name.",
        )
    return _tagged(
        "custom",
        python_type=python_type,
        artifact_dir=artifact_relpath,
        lazy=lazy,
        payload=payload,
    )


def _load_external_node(
    node: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    config: ResultConfig,
    logical_path: LogicalPath,
) -> Any:
    codec_id = _require_str(node, "codec", logical_path)
    _resolve_artifact_dir(
        bundle_dir, _require_str(node, "artifact_dir", logical_path), logical_path
    )
    lazy = _require_bool(node, "lazy", logical_path)
    meta = node.get("meta")
    if lazy:
        return FuruLazy(
            _loader=lambda: _materialize_external_node(
                node,
                bundle_dir=bundle_dir,
                config=config,
                logical_path=logical_path,
            ),
            _logical_path=logical_path,
            _descriptor=f"codec={codec_id}",
            _meta=meta,
        )
    return _materialize_external_node(
        node,
        bundle_dir=bundle_dir,
        config=config,
        logical_path=logical_path,
    )


def _materialize_external_node(
    node: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    config: ResultConfig,
    logical_path: LogicalPath,
) -> Any:
    codec_id = _require_str(node, "codec", logical_path)
    artifact_relpath = _require_str(node, "artifact_dir", logical_path)
    artifact_dir = _resolve_artifact_dir(bundle_dir, artifact_relpath, logical_path)
    codec = config.registry.get_codec(codec_id)
    load_ctx = LoadContext(
        bundle_dir=bundle_dir,
        artifact_dir=artifact_dir,
        node=node,
        logical_path=logical_path,
        registry=config.registry,
    )
    try:
        return codec.load(load_ctx)
    except ResultCodecError:
        raise
    except Exception as exc:
        raise ResultLoadError(
            logical_path,
            f"codec {codec_id!r} failed while loading from {artifact_relpath}: {exc}",
        ) from exc


def _load_custom_node(
    node: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    config: ResultConfig,
    logical_path: LogicalPath,
) -> Any:
    python_type = _require_str(node, "python_type", logical_path)
    cls = _import_type(python_type, logical_path)
    _resolve_artifact_dir(
        bundle_dir, _require_str(node, "artifact_dir", logical_path), logical_path
    )
    lazy = _require_bool(node, "lazy", logical_path)
    payload = node.get("payload")
    if lazy:
        return FuruLazy(
            _loader=lambda: _materialize_custom_node(
                cls,
                payload,
                node,
                bundle_dir=bundle_dir,
                config=config,
                logical_path=logical_path,
            ),
            _logical_path=logical_path,
            _descriptor=f"kind=custom type={python_type}",
            _meta=payload,
        )
    return _materialize_custom_node(
        cls,
        payload,
        node,
        bundle_dir=bundle_dir,
        config=config,
        logical_path=logical_path,
    )


def _materialize_custom_node(
    cls: type[Any],
    payload: JsonValue | None,
    node: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    config: ResultConfig,
    logical_path: LogicalPath,
) -> Any:
    artifact_relpath = _require_str(node, "artifact_dir", logical_path)
    artifact_dir = _resolve_artifact_dir(bundle_dir, artifact_relpath, logical_path)
    load_ctx = LoadContext(
        bundle_dir=bundle_dir,
        artifact_dir=artifact_dir,
        node=node,
        logical_path=logical_path,
        registry=config.registry,
    )
    load_method = getattr(cls, "__furu_result_load__", None)
    if not callable(load_method):
        raise ResultLoadError(
            logical_path,
            f"{cls.__module__}.{cls.__qualname__} does not define __furu_result_load__.",
        )
    try:
        return load_method(payload, load_ctx)
    except Exception as exc:
        raise ResultLoadError(
            logical_path,
            f"custom result protocol failed while loading {cls.__module__}.{cls.__qualname__}: {exc}",
        ) from exc


def _dump_mapping(value: dict[Any, Any], ctx: DumpNodeContext) -> JsonValue:
    if any(not isinstance(key, str) for key in value):
        raise ResultSerializationError(
            ctx.logical_path,
            dict,
            "only string dictionary keys are supported by the default JSON result format.",
        )

    if _WRAPPER_KEY in value:
        return _tagged(
            "mapping",
            items=[
                {
                    "key": key,
                    "value": dump_node(item, ctx.child(key)),
                }
                for key, item in value.items()
            ],
        )

    return {key: dump_node(item, ctx.child(key)) for key, item in value.items()}


def _dump_dataclass(value: Any, ctx: DumpNodeContext) -> JsonValue:
    hints = get_type_hints(type(value), include_extras=True)
    try:
        python_type = fully_qualified_name(type(value))
    except ValueError as exc:
        raise ResultSerializationError(
            ctx.logical_path,
            type(value),
            str(exc),
        ) from exc

    fields_payload = {
        field.name: dump_node(
            getattr(value, field.name),
            ctx.child(field.name),
            field_spec=_spec_from_annotation(hints.get(field.name)),
        )
        for field in dataclasses.fields(value)
    }
    return _tagged("dataclass", python_type=python_type, fields=fields_payload)


def _dump_pydantic(value: BaseModel, ctx: DumpNodeContext) -> JsonValue:
    hints = get_type_hints(type(value), include_extras=True)
    try:
        python_type = fully_qualified_name(type(value))
    except ValueError as exc:
        raise ResultSerializationError(
            ctx.logical_path,
            type(value),
            str(exc),
        ) from exc

    fields_payload = {
        field_name: dump_node(
            getattr(value, field_name),
            ctx.child(field_name),
            field_spec=_spec_from_annotation(hints.get(field_name)),
        )
        for field_name in type(value).model_fields
    }
    return _tagged("pydantic", python_type=python_type, fields=fields_payload)


def _dump_sorted_set_items(
    value: set[Any] | frozenset[Any], ctx: DumpNodeContext
) -> list[JsonValue]:
    dumped = [dump_node(item, ctx.child(i)) for i, item in enumerate(value)]
    return sorted(dumped, key=_stable_json_dump)


def _spec_from_annotation(annotation: Any) -> ResultSpec | None:
    if annotation is None or get_origin(annotation) is not Annotated:
        return None
    spec: ResultSpec | None = None
    _, *metadata = get_args(annotation)
    for item in metadata:
        if isinstance(item, SaveWith):
            spec = merge_specs(spec, item.spec)
    return spec


def _supports_custom_protocol(value: Any) -> bool:
    return callable(getattr(value, "__furu_result_dump__", None)) and callable(
        getattr(type(value), "__furu_result_load__", None)
    )


def _resolve_codec_ref(
    config: ResultConfig,
    codec_ref: str | ResultCodec[Any],
) -> ResultCodec[Any]:
    if isinstance(codec_ref, str):
        return config.registry.get_codec(codec_ref)
    config.registry.register_codec(codec_ref)
    return codec_ref


def _maybe_python_type(tp: type[Any]) -> str | None:
    try:
        return fully_qualified_name(tp)
    except ValueError:
        return None


def _is_dataclass_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _is_json_compatible(value: Any, *, allow_reserved_key: bool) -> bool:
    match value:
        case None | bool() | int() | str():
            return True
        case float():
            return math.isfinite(value)
        case list():
            return all(
                _is_json_compatible(item, allow_reserved_key=allow_reserved_key)
                for item in value
            )
        case dict():
            for key, item in value.items():
                if not isinstance(key, str):
                    return False
                if not allow_reserved_key and key == _WRAPPER_KEY:
                    return False
                if not _is_json_compatible(item, allow_reserved_key=allow_reserved_key):
                    return False
            return True
        case _:
            return False


@dataclass
class _CycleGuard:
    value: Any
    ctx: DumpNodeContext

    def __enter__(self) -> None:
        object_id = id(self.value)
        if object_id in self.ctx.active_ids:
            raise ResultSerializationError(
                self.ctx.logical_path,
                type(self.value),
                "cycles are not supported in Furu results.",
            )
        self.ctx.active_ids.add(object_id)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.ctx.active_ids.discard(id(self.value))


def _cycle_guard(value: Any, ctx: DumpNodeContext) -> _CycleGuard:
    return _CycleGuard(value=value, ctx=ctx)


def _tagged(kind: str, **payload: JsonValue) -> JsonValue:
    wrapped: dict[str, JsonValue] = {"kind": kind}
    wrapped.update(payload)
    return _tagged_dict(wrapped)


def _tagged_dict(payload: dict[str, JsonValue]) -> JsonValue:
    return {_WRAPPER_KEY: payload}


def _import_type(qualified_name: str, logical_path: LogicalPath) -> type[Any]:
    try:
        module_name, class_name = qualified_name.rsplit(".", 1)
        module = import_module(module_name)
        value = getattr(module, class_name)
    except Exception as exc:
        raise ResultLoadError(
            logical_path,
            f"could not import Python type {qualified_name!r}: {exc}",
        ) from exc
    if not isinstance(value, type):
        raise ResultLoadError(
            logical_path, f"{qualified_name!r} does not resolve to a type."
        )
    return value


def _resolve_artifact_dir(
    bundle_dir: Path, artifact_relpath: str, logical_path: LogicalPath
) -> Path:
    artifact_path = Path(artifact_relpath)
    if artifact_path.is_absolute():
        raise ResultLoadError(
            logical_path, "artifact_dir must be relative to the result bundle."
        )
    bundle_root = bundle_dir.resolve(strict=False)
    target = (bundle_root / artifact_path).resolve(strict=False)
    if os.path.commonpath([os.fspath(bundle_root), os.fspath(target)]) != os.fspath(
        bundle_root
    ):
        raise ResultLoadError(logical_path, "artifact_dir escapes the result bundle.")
    return target


def _require_str(
    node: dict[str, JsonValue], key: str, logical_path: LogicalPath
) -> str:
    value = node.get(key)
    if not isinstance(value, str):
        raise ResultLoadError(logical_path, f"expected string field {key!r}.")
    return value


def _require_bool(
    node: dict[str, JsonValue], key: str, logical_path: LogicalPath
) -> bool:
    value = node.get(key)
    if not isinstance(value, bool):
        raise ResultLoadError(logical_path, f"expected boolean field {key!r}.")
    return value


def _require_dict(
    node: dict[str, JsonValue], key: str, logical_path: LogicalPath
) -> dict[str, JsonValue]:
    value = node.get(key)
    if not isinstance(value, dict):
        raise ResultLoadError(logical_path, f"expected object field {key!r}.")
    return value


def _require_list(
    node: dict[str, JsonValue], key: str, logical_path: LogicalPath
) -> list[JsonValue]:
    value = node.get(key)
    if not isinstance(value, list):
        raise ResultLoadError(logical_path, f"expected list field {key!r}.")
    return value
