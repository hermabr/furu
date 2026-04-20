from __future__ import annotations

import dataclasses
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, cast, get_type_hints

from pydantic import BaseModel

from furu.results.api import (
    ResultConfig,
    ResultPolicy,
    extract_save_with,
    merge_policies,
    supports_furu_result_protocol,
    unwrap_result_value,
)
from furu.results.codecs import DumpContext, LoadContext, ResultCodec
from furu.results.errors import (
    ResultDeserializationError,
    ResultPathCollisionError,
    ResultSerializationError,
)
from furu.results.lazy import LazyValue
from furu.results.nodes import (
    get_furu_payload,
    is_external_node,
    is_furu_node,
    make_furu_node,
)
from furu.results.paths import LogicalPath
from furu.utils import JsonValue, fully_qualified_name, import_fully_qualified_name


@dataclass(slots=True)
class _DumpState:
    active_ids: set[int] = field(default_factory=set)
    artifact_paths: dict[str, LogicalPath] = field(default_factory=dict)


def dump_root(value: object, *, bundle_dir, config: ResultConfig) -> JsonValue:
    ctx = DumpContext(
        bundle_dir=bundle_dir,
        artifact_dir=bundle_dir / LogicalPath().artifact_relative_dir(),
        logical_path=LogicalPath(),
        registry=config.registry,
        config=config,
    )
    return _dump_node(value, ctx, _DumpState(), field_policy=ResultPolicy())


def load_root(node: JsonValue, *, bundle_dir, config: ResultConfig) -> object:
    ctx = LoadContext(
        bundle_dir=bundle_dir,
        artifact_dir=bundle_dir / LogicalPath().artifact_relative_dir(),
        logical_path=LogicalPath(),
        registry=config.registry,
        config=config,
        node=None,
    )
    return _load_node(node, ctx)


def _is_json_scalar(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _dump_json_scalar(value: object, logical_path: LogicalPath) -> JsonValue:
    if not _is_json_scalar(value):
        raise TypeError("Expected JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ResultSerializationError(
            "Non-finite floats are not valid JSON values",
            logical_path=logical_path,
        )
    return cast(JsonValue, value)


def _supports_json_file_codec(value: object) -> bool:
    try:
        _json_only(value, LogicalPath())
    except ResultSerializationError:
        return False
    return True


def _json_only(value: object, logical_path: LogicalPath) -> JsonValue:
    if _is_json_scalar(value):
        return _dump_json_scalar(value, logical_path)
    if isinstance(value, list):
        return [
            _json_only(item, logical_path.child_index(index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        converted: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResultSerializationError(
                    "JSON-compatible dict keys must be strings",
                    logical_path=logical_path,
                )
            converted[key] = _json_only(item, logical_path.child_key(key))
        return converted
    raise ResultSerializationError(
        "Value is not JSON-compatible",
        logical_path=logical_path,
    )


@contextmanager
def _track_object(value: object, logical_path: LogicalPath, state: _DumpState):
    if not _should_track_cycle(value):
        yield
        return

    object_id = id(value)
    if object_id in state.active_ids:
        raise ResultSerializationError(
            "Cycles are not supported in result persistence",
            logical_path=logical_path,
        )
    state.active_ids.add(object_id)
    try:
        yield
    finally:
        state.active_ids.remove(object_id)


def _should_track_cycle(value: object) -> bool:
    return (
        isinstance(value, (list, dict, tuple, set, frozenset, BaseModel))
        or dataclasses.is_dataclass(value)
        and not isinstance(value, type)
    )


def _resolve_codec_reference(
    codec_ref: str | ResultCodec[Any],
    ctx: DumpContext,
) -> ResultCodec[Any]:
    if isinstance(codec_ref, str):
        return ctx.registry.get_codec(codec_ref, logical_path=ctx.logical_path)
    ctx.registry.register_codec(codec_ref)
    return codec_ref


def _resolve_codec_or_json_file(value: object, ctx: DumpContext) -> ResultCodec[Any]:
    if supports_furu_result_protocol(value):
        return ctx.registry.get_codec(
            "furu.object-protocol.v1",
            logical_path=ctx.logical_path,
        )
    codec = ctx.registry.resolve_type(value)
    if codec is not None:
        return codec
    if _supports_json_file_codec(value):
        return ctx.registry.get_codec("furu.json.v1", logical_path=ctx.logical_path)
    raise ResultSerializationError(
        (
            f"Cannot externalize value of type {type(value).__module__}.{type(value).__qualname__}. "
            "Wrap it with furu.result(...), register a ResultCodec in _result_config(), "
            "or implement FuruResult."
        ),
        logical_path=ctx.logical_path,
    )


def _reserve_artifact_dir(ctx: DumpContext, state: _DumpState) -> None:
    artifact_dir = ctx.logical_path.artifact_relative_dir().as_posix()
    previous = state.artifact_paths.get(artifact_dir)
    if previous is not None and previous != ctx.logical_path:
        raise ResultPathCollisionError(artifact_dir, ctx.logical_path, previous)
    state.artifact_paths[artifact_dir] = ctx.logical_path


def _dump_external(
    value: object,
    codec: ResultCodec[Any],
    ctx: DumpContext,
    state: _DumpState,
    *,
    lazy: bool,
    python_type: str | None = None,
) -> dict[str, JsonValue]:
    _reserve_artifact_dir(ctx, state)
    meta = codec.dump(value, ctx)
    payload: dict[str, JsonValue] = {
        "artifact_dir": ctx.logical_path.artifact_relative_dir().as_posix(),
        "codec": codec.codec_id,
        "kind": "external",
        "lazy": lazy,
    }
    if python_type is not None:
        payload["python_type"] = python_type
    if meta is not None:
        payload["meta"] = meta
    payload_without_kind = dict(payload)
    del payload_without_kind["kind"]
    return make_furu_node("external", **payload_without_kind)


def _dump_mapping(
    value: dict[Any, Any],
    ctx: DumpContext,
    state: _DumpState,
) -> JsonValue:
    items: list[list[JsonValue]] = []
    mapping: dict[str, JsonValue] = {}
    has_reserved_key = False

    for key, item in value.items():
        if not isinstance(key, str):
            raise ResultSerializationError(
                f"Result dict keys must be strings; got {key!r}",
                logical_path=ctx.logical_path,
            )
        has_reserved_key = has_reserved_key or key == "$furu"
        child_node = _dump_node(
            item, ctx.child_key(key), state, field_policy=ResultPolicy()
        )
        mapping[key] = child_node
        items.append([key, child_node])

    if has_reserved_key:
        return make_furu_node("mapping", items=cast(JsonValue, items))
    return mapping


def _dump_sequence_items(
    items: list[Any] | tuple[Any, ...],
    ctx: DumpContext,
    state: _DumpState,
) -> list[JsonValue]:
    return [
        _dump_node(item, ctx.child_index(index), state, field_policy=ResultPolicy())
        for index, item in enumerate(items)
    ]


def _ensure_importable_type(tp: type[object], logical_path: LogicalPath) -> str:
    try:
        return fully_qualified_name(tp)
    except ValueError as exc:
        raise ResultSerializationError(
            (
                f"Result type {tp.__module__}.{tp.__qualname__} is not importable. "
                "Move it to a module-level class so the result bundle can be loaded."
            ),
            logical_path=logical_path,
        ) from exc


@lru_cache(maxsize=None)
def _type_hints_with_extras(tp: type[object]) -> dict[str, object]:
    return get_type_hints(tp, include_extras=True)


def _dump_dataclass(
    value: Any,
    ctx: DumpContext,
    state: _DumpState,
) -> dict[str, JsonValue]:
    field_nodes: dict[str, JsonValue] = {}
    type_hints = _type_hints_with_extras(type(value))
    for dc_field in dataclasses.fields(value):
        child_ctx = ctx.child_field(dc_field.name)
        field_policy = extract_save_with(type_hints.get(dc_field.name))
        field_nodes[dc_field.name] = _dump_node(
            getattr(value, dc_field.name),
            child_ctx,
            state,
            field_policy=field_policy,
        )
    return make_furu_node(
        "dataclass",
        python_type=_ensure_importable_type(type(value), ctx.logical_path),
        fields=field_nodes,
    )


def _dump_pydantic(
    value: BaseModel,
    ctx: DumpContext,
    state: _DumpState,
) -> dict[str, JsonValue]:
    field_nodes: dict[str, JsonValue] = {}
    type_hints = _type_hints_with_extras(type(value))
    for field_name in type(value).model_fields:
        child_ctx = ctx.child_field(field_name)
        field_policy = extract_save_with(type_hints.get(field_name))
        field_nodes[field_name] = _dump_node(
            getattr(value, field_name),
            child_ctx,
            state,
            field_policy=field_policy,
        )
    return make_furu_node(
        "pydantic",
        python_type=_ensure_importable_type(type(value), ctx.logical_path),
        fields=field_nodes,
    )


def _dump_set_like(
    value: set[Any] | frozenset[Any],
    kind: str,
    ctx: DumpContext,
    state: _DumpState,
) -> dict[str, JsonValue]:
    dumped_items: list[JsonValue] = []
    for index, item in enumerate(sorted(value, key=repr)):
        child_node = _dump_node(
            item,
            ctx.child_index(index),
            state,
            field_policy=ResultPolicy(),
        )
        if is_external_node(child_node):
            raise ResultSerializationError(
                "Externalized leaves inside sets and frozensets are not supported yet",
                logical_path=ctx.logical_path,
            )
        dumped_items.append(child_node)
    return make_furu_node(kind, items=dumped_items)


def _dump_node(
    value: object,
    ctx: DumpContext,
    state: _DumpState,
    *,
    field_policy: ResultPolicy,
) -> JsonValue:
    wrapper_policy, unwrapped = unwrap_result_value(value)
    rule_policy = ctx.config.matching_policy(ctx.logical_path, unwrapped)
    policy = merge_policies(field_policy, rule_policy, wrapper_policy)

    if policy.lazy is True:
        codec = (
            _resolve_codec_reference(policy.codec, ctx)
            if policy.codec is not None
            else _resolve_codec_or_json_file(unwrapped, ctx)
        )
        python_type = None
        if codec.codec_id == "furu.object-protocol.v1":
            python_type = _ensure_importable_type(type(unwrapped), ctx.logical_path)
        return _dump_external(
            unwrapped,
            codec,
            ctx,
            state,
            lazy=True,
            python_type=python_type,
        )

    if policy.codec is not None:
        codec = _resolve_codec_reference(policy.codec, ctx)
        python_type = None
        if codec.codec_id == "furu.object-protocol.v1":
            python_type = _ensure_importable_type(type(unwrapped), ctx.logical_path)
        return _dump_external(
            unwrapped,
            codec,
            ctx,
            state,
            lazy=bool(policy.lazy),
            python_type=python_type,
        )

    if supports_furu_result_protocol(unwrapped):
        codec = ctx.registry.get_codec(
            "furu.object-protocol.v1",
            logical_path=ctx.logical_path,
        )
        return _dump_external(
            unwrapped,
            codec,
            ctx,
            state,
            lazy=bool(policy.lazy),
            python_type=_ensure_importable_type(type(unwrapped), ctx.logical_path),
        )

    codec = ctx.registry.resolve_type(unwrapped)
    if codec is not None:
        return _dump_external(
            unwrapped,
            codec,
            ctx,
            state,
            lazy=bool(policy.lazy),
        )

    if _is_json_scalar(unwrapped):
        return _dump_json_scalar(unwrapped, ctx.logical_path)

    with _track_object(unwrapped, ctx.logical_path, state):
        if isinstance(unwrapped, list):
            return _dump_sequence_items(cast(list[Any], unwrapped), ctx, state)

        if isinstance(unwrapped, dict):
            return _dump_mapping(cast(dict[Any, Any], unwrapped), ctx, state)

        if isinstance(unwrapped, tuple):
            return make_furu_node(
                "tuple",
                items=cast(JsonValue, _dump_sequence_items(unwrapped, ctx, state)),
            )

        if isinstance(unwrapped, set):
            return _dump_set_like(cast(set[Any], unwrapped), "set", ctx, state)

        if isinstance(unwrapped, frozenset):
            return _dump_set_like(
                cast(frozenset[Any], unwrapped),
                "frozenset",
                ctx,
                state,
            )

        if dataclasses.is_dataclass(unwrapped) and not isinstance(unwrapped, type):
            return _dump_dataclass(unwrapped, ctx, state)

        if isinstance(unwrapped, BaseModel):
            return _dump_pydantic(unwrapped, ctx, state)

    raise ResultSerializationError(
        (
            f"Cannot serialize value of type {type(unwrapped).__module__}.{type(unwrapped).__qualname__}. "
            "Wrap it with furu.result(...), register a ResultCodec in _result_config(), "
            "or implement FuruResult."
        ),
        logical_path=ctx.logical_path,
    )


def _load_fields(
    fields_node: object,
    ctx: LoadContext,
) -> dict[str, object]:
    if not isinstance(fields_node, dict):
        raise ResultDeserializationError(
            "Structured result fields must be a JSON object",
            logical_path=ctx.logical_path,
        )
    result: dict[str, object] = {}
    for name, child in fields_node.items():
        if not isinstance(name, str):
            raise ResultDeserializationError(
                "Structured result field names must be strings",
                logical_path=ctx.logical_path,
            )
        result[name] = _load_node(cast(JsonValue, child), ctx.child_field(name))
    return result


def _load_external(payload: dict[str, JsonValue], ctx: LoadContext) -> object:
    codec_id = payload.get("codec")
    artifact_dir = payload.get("artifact_dir")
    lazy = payload.get("lazy")
    if not isinstance(codec_id, str):
        raise ResultDeserializationError(
            "External result nodes require codec",
            logical_path=ctx.logical_path,
        )
    if not isinstance(artifact_dir, str):
        raise ResultDeserializationError(
            "External result nodes require artifact_dir",
            logical_path=ctx.logical_path,
        )
    if not isinstance(lazy, bool):
        raise ResultDeserializationError(
            "External result nodes require boolean lazy",
            logical_path=ctx.logical_path,
        )

    codec = ctx.registry.get_codec(codec_id, logical_path=ctx.logical_path)
    child_ctx = ctx.for_external(
        artifact_dir=ctx.bundle_dir / artifact_dir,
        node=payload,
    )
    meta = payload.get("meta")
    if lazy:
        return LazyValue(codec, child_ctx, meta)
    return codec.load(child_ctx, meta)


def _load_mapping_items(
    items: object,
    ctx: LoadContext,
) -> dict[str, object]:
    if not isinstance(items, list):
        raise ResultDeserializationError(
            "Mapping escape items must be a list",
            logical_path=ctx.logical_path,
        )
    result: dict[str, object] = {}
    for raw_item in items:
        if (
            not isinstance(raw_item, list)
            or len(raw_item) != 2
            or not isinstance(raw_item[0], str)
        ):
            raise ResultDeserializationError(
                "Mapping escape items must be [str, value] pairs",
                logical_path=ctx.logical_path,
            )
        key = raw_item[0]
        result[key] = _load_node(cast(JsonValue, raw_item[1]), ctx.child_key(key))
    return result


def _load_node(node: JsonValue, ctx: LoadContext) -> object:
    if _is_json_scalar(node):
        return node

    if isinstance(node, list):
        return [
            _load_node(item, ctx.child_index(index)) for index, item in enumerate(node)
        ]

    if isinstance(node, dict) and not is_furu_node(node):
        return {
            key: _load_node(value, ctx.child_key(key)) for key, value in node.items()
        }

    payload = get_furu_payload(node, logical_path=ctx.logical_path)
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise ResultDeserializationError(
            "Furu manifest nodes require kind",
            logical_path=ctx.logical_path,
        )

    if kind == "external":
        return _load_external(payload, ctx)

    if kind == "dataclass":
        python_type = payload.get("python_type")
        if not isinstance(python_type, str):
            raise ResultDeserializationError(
                "Dataclass nodes require python_type",
                logical_path=ctx.logical_path,
            )
        cls = import_fully_qualified_name(python_type)
        return cls(**_load_fields(payload.get("fields"), ctx))

    if kind == "pydantic":
        python_type = payload.get("python_type")
        if not isinstance(python_type, str):
            raise ResultDeserializationError(
                "Pydantic nodes require python_type",
                logical_path=ctx.logical_path,
            )
        cls = import_fully_qualified_name(python_type)
        model_construct = getattr(cls, "model_construct", None)
        if not callable(model_construct):
            raise ResultDeserializationError(
                "Pydantic result types must define model_construct()",
                logical_path=ctx.logical_path,
            )
        return model_construct(**_load_fields(payload.get("fields"), ctx))

    if kind == "tuple":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ResultDeserializationError(
                "Tuple nodes require items",
                logical_path=ctx.logical_path,
            )
        return tuple(
            _load_node(item, ctx.child_index(index)) for index, item in enumerate(items)
        )

    if kind == "set":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ResultDeserializationError(
                "Set nodes require items",
                logical_path=ctx.logical_path,
            )
        return set(
            _load_node(item, ctx.child_index(index)) for index, item in enumerate(items)
        )

    if kind == "frozenset":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ResultDeserializationError(
                "Frozenset nodes require items",
                logical_path=ctx.logical_path,
            )
        return frozenset(
            _load_node(item, ctx.child_index(index)) for index, item in enumerate(items)
        )

    if kind == "mapping":
        return _load_mapping_items(payload.get("items"), ctx)

    raise ResultDeserializationError(
        f"Unknown Furu manifest node kind {kind!r}",
        logical_path=ctx.logical_path,
    )
