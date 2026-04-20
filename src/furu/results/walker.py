from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast, get_type_hints

from pydantic import BaseModel

from furu.results.context import DumpContext, LoadContext
from furu.results.lazy import LazyValue
from furu.results.nodes import (
    DataclassNode,
    ExternalNode,
    FrozenSetNode,
    ManifestValue,
    MappingNode,
    make_dataclass_node,
    make_frozenset_node,
    make_mapping_node,
    make_protocol_node,
    make_pydantic_node,
    make_set_node,
    make_tuple_node,
    ProtocolNode,
    PydanticNode,
    SetNode,
    TupleNode,
    unwrap_node,
)
from furu.results.paths import LogicalPath
from furu.results.protocol import (
    protocol_type_name,
    resolve_type_name,
    supports_furu_result_protocol,
)
from furu.results.registry import ResultCodec
from furu.results.rules import (
    SaveSpec,
    _WrappedValue,
    annotation_save_spec,
    merge_specs,
    path_rule_spec,
    type_rule_spec,
)
from furu.utils import JsonValue, class_label, fully_qualified_name


class ResultSerializationError(TypeError):
    pass


class ResultDeserializationError(TypeError):
    pass


class ResultCycleError(ResultSerializationError):
    pass


def dump_manifest_root(
    value: object,
    *,
    result_dir: Path,
    config,
) -> ManifestValue:
    active_ids: set[int] = set()

    def dump_value(
        current: object,
        ctx: DumpContext,
        field_annotation: object | None,
        force_current_inline: bool,
    ) -> ManifestValue:
        explicit_spec: SaveSpec | None = None
        if isinstance(current, _WrappedValue):
            explicit_spec = current.spec
            current = current.value

        if _is_scalar(current):
            return cast(ManifestValue, current)

        cycle_id = _cycle_id(current)
        added_cycle_id = False
        if cycle_id is not None:
            if cycle_id in active_ids and not force_current_inline:
                raise ResultCycleError(
                    f"Cycle detected while saving {ctx.path.display()} ({type(current).__name__})"
                )
            if cycle_id not in active_ids:
                active_ids.add(cycle_id)
                added_cycle_id = True

        try:
            return _dump_non_scalar(
                current,
                ctx,
                field_annotation=field_annotation,
                explicit_spec=explicit_spec,
                force_current_inline=force_current_inline,
            )
        finally:
            if cycle_id is not None and added_cycle_id:
                active_ids.remove(cycle_id)

    root_ctx = DumpContext(
        result_dir=result_dir,
        config=config,
        path=LogicalPath(),
        _dump_value=dump_value,
    )
    return dump_value(value, root_ctx, None, False)


def load_manifest_root(
    node: ManifestValue,
    *,
    result_dir: Path,
    config,
) -> object:
    def load_value(current: ManifestValue, ctx: LoadContext) -> object:
        tagged = unwrap_node(current)
        if tagged is None:
            if isinstance(current, list):
                return [
                    load_value(item, ctx.child(index))
                    for index, item in enumerate(current)
                ]
            if isinstance(current, dict):
                return {
                    key: load_value(value, ctx.child(key))
                    for key, value in current.items()
                }
            return current

        kind = tagged["kind"]
        if kind == "external":
            return _load_external(cast(ExternalNode, tagged), ctx)
        if kind == "dataclass":
            dataclass_node = cast(DataclassNode, tagged)
            tp = _require_type(kind, dataclass_node["python_type"])
            values = {
                name: load_value(value, ctx.child(name))
                for name, value in dataclass_node["fields"].items()
            }
            return tp(**values)
        if kind == "pydantic":
            pydantic_node = cast(PydanticNode, tagged)
            tp = _require_pydantic_type(pydantic_node["python_type"])
            values = {
                name: load_value(value, ctx.child(name))
                for name, value in pydantic_node["fields"].items()
            }
            return tp.model_construct(**cast(dict[str, Any], values))
        if kind == "tuple":
            tuple_node = cast(TupleNode, tagged)
            return tuple(
                load_value(item, ctx.child(index))
                for index, item in enumerate(tuple_node["items"])
            )
        if kind == "set":
            set_node = cast(SetNode, tagged)
            return {
                load_value(item, ctx.child(index))
                for index, item in enumerate(set_node["items"])
            }
        if kind == "frozenset":
            frozenset_node = cast(FrozenSetNode, tagged)
            return frozenset(
                load_value(item, ctx.child(index))
                for index, item in enumerate(frozenset_node["items"])
            )
        if kind == "mapping":
            mapping_node = cast(MappingNode, tagged)
            return {
                key: load_value(value, ctx.child(key))
                for key, value in mapping_node["items"].items()
            }
        if kind == "protocol":
            protocol_node = cast(ProtocolNode, tagged)
            tp = _require_type(kind, protocol_node["python_type"])
            load_method = getattr(tp, "__furu_load_result__", None)
            if not callable(load_method):
                raise ResultDeserializationError(
                    f"{ctx.path.display()} cannot load protocol result for {protocol_node['python_type']}"
                )
            return load_method(protocol_node["value"], ctx)
        raise ResultDeserializationError(f"Unknown manifest node kind {kind!r}")

    root_ctx = LoadContext(
        result_dir=result_dir,
        config=config,
        path=LogicalPath(),
        _load_value=load_value,
    )
    return load_value(node, root_ctx)


def _dump_non_scalar(
    value: object,
    ctx: DumpContext,
    *,
    field_annotation: object | None,
    explicit_spec: SaveSpec | None,
    force_current_inline: bool,
) -> ManifestValue:
    resolved = _resolve_current_handler(
        value,
        ctx=ctx,
        field_annotation=field_annotation,
        explicit_spec=explicit_spec,
    )

    if not force_current_inline:
        if resolved.codec is not None:
            if resolved.codec.codec_id == "json_file":
                inline = ctx.dump(value, annotation=field_annotation, inline=True)
                return _externalize_inline(
                    inline,
                    ctx=ctx,
                    lazy=bool(resolved.spec.lazy),
                    python_type=class_label(type(value)),
                )
            dumped = resolved.codec.dump(value, ctx, resolved.spec)
            return _with_python_type(dumped, class_label(type(value)))

        if resolved.spec.lazy:
            inline = ctx.dump(value, annotation=field_annotation, inline=True)
            return _externalize_inline(
                inline,
                ctx=ctx,
                lazy=True,
                python_type=class_label(type(value)),
            )

    if resolved.use_protocol:
        save_method = getattr(value, "__furu_save_result__")
        payload = cast(JsonValue, save_method(ctx))
        return make_protocol_node(protocol_type_name(type(value)), payload)

    if isinstance(value, list):
        return [ctx.dump(item, token=index) for index, item in enumerate(value)]

    if isinstance(value, dict):
        items: dict[str, ManifestValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _unsupported_key_error(ctx, key)
            items[key] = ctx.dump(item, token=key)
        if "$furu" in items:
            return make_mapping_node(items)
        return items

    if isinstance(value, tuple):
        return make_tuple_node(
            [ctx.dump(item, token=index) for index, item in enumerate(value)]
        )

    if isinstance(value, set):
        return make_set_node(_dump_set_items(value, ctx))

    if isinstance(value, frozenset):
        return make_frozenset_node(_dump_set_items(value, ctx))

    if is_dataclass(value) and not isinstance(value, type):
        type_hints = _type_hints(type(value))
        return make_dataclass_node(
            fully_qualified_name(type(value)),
            {
                field.name: ctx.dump(
                    getattr(value, field.name),
                    token=field.name,
                    annotation=type_hints.get(field.name),
                )
                for field in fields(value)
            },
        )

    if isinstance(value, BaseModel):
        type_hints = _type_hints(type(value))
        return make_pydantic_node(
            fully_qualified_name(type(value)),
            {
                name: ctx.dump(
                    getattr(value, name),
                    token=name,
                    annotation=type_hints.get(name),
                )
                for name in type(value).model_fields
            },
        )

    raise ResultSerializationError(_unsupported_value_message(ctx, value))


def _resolve_current_handler(
    value: object,
    *,
    ctx: DumpContext,
    field_annotation: object | None,
    explicit_spec: SaveSpec | None,
) -> "_ResolvedHandler":
    annotation_spec = annotation_save_spec(field_annotation)
    path_spec = path_rule_spec(ctx.config, ctx.path)
    high_spec = merge_specs(annotation_spec, path_spec, explicit_spec)

    if high_spec.serializer is not None:
        return _ResolvedHandler(
            spec=high_spec,
            codec=ctx.config.registry.require(high_spec.serializer),
            use_protocol=False,
        )

    if supports_furu_result_protocol(value):
        return _ResolvedHandler(spec=high_spec, codec=None, use_protocol=True)

    low_spec = type_rule_spec(ctx.config, value)
    spec = merge_specs(low_spec, high_spec)
    codec = (
        ctx.config.registry.require(spec.serializer)
        if spec.serializer is not None
        else ctx.config.registry.codec_for_value(value)
    )
    return _ResolvedHandler(spec=spec, codec=codec, use_protocol=False)


@cache
def _type_hints(tp: type[object]) -> dict[str, object]:
    return get_type_hints(tp, include_extras=True)


def _externalize_inline(
    value: ManifestValue,
    *,
    ctx: DumpContext,
    lazy: bool,
    python_type: str,
) -> ManifestValue:
    codec = ctx.config.registry.require("json_file")
    dumped = codec.dump(value, ctx, SaveSpec(lazy=lazy))
    return _with_python_type(dumped, python_type)


def _with_python_type(node: ManifestValue, python_type: str) -> ManifestValue:
    tagged = unwrap_node(node)
    if tagged is None or tagged["kind"] != "external":
        return node
    tagged["python_type"] = python_type
    return node


def _load_external(node: ExternalNode, ctx: LoadContext) -> object:
    codec = ctx.config.registry.require(node["serializer"])

    def loader() -> object:
        return codec.load(cast(ManifestValue, node), ctx)

    if node["lazy"]:
        return LazyValue(
            loader,
            _path=ctx.path,
            _serializer=node["serializer"],
            _meta=node.get("meta"),
            _python_type=node["python_type"],
        )
    return loader()


def _dump_set_items(
    values: set[Any] | frozenset[Any], ctx: DumpContext
) -> list[ManifestValue]:
    items = [ctx.dump(item, token=index) for index, item in enumerate(values)]
    return sorted(items, key=_stable_manifest_json)


def _stable_manifest_json(value: ManifestValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, bool | int | float | str)


def _cycle_id(value: object) -> int | None:
    if _is_scalar(value):
        return None
    return id(value)


def _unsupported_key_error(ctx: DumpContext, key: object) -> ResultSerializationError:
    return ResultSerializationError(
        f"{ctx.path.display()} contains unsupported mapping key {key!r} ({type(key).__name__}); "
        "convert keys to strings or wrap the value with save_with(...)."
    )


def _unsupported_value_message(ctx: DumpContext, value: object) -> str:
    return (
        f"{ctx.path.display()} cannot serialize value of type {type(value).__name__}; "
        "wrap it with save_with(...), register a codec, or convert it to a supported structural type."
    )


def _require_type(kind: str, type_name: str) -> type[object]:
    tp = resolve_type_name(type_name)
    if not isinstance(tp, type):
        raise ResultDeserializationError(
            f"Manifest {kind} node references non-type {type_name!r}"
        )
    return tp


def _require_pydantic_type(type_name: str) -> type[BaseModel]:
    tp = _require_type("pydantic", type_name)
    if not issubclass(tp, BaseModel):
        raise ResultDeserializationError(
            f"Manifest pydantic node references non-Pydantic type {type_name!r}"
        )
    return tp


@dataclass(frozen=True, slots=True)
class _ResolvedHandler:
    spec: SaveSpec
    codec: ResultCodec[Any] | None
    use_protocol: bool
