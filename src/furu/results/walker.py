from __future__ import annotations

import importlib
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast, get_type_hints

from pydantic import BaseModel as PydanticBaseModel

from furu.results.lazy import LazyValue
from furu.results.nodes import (
    DataclassNode,
    ExternalNode,
    ManifestNode,
    MappingItem,
    MappingNode,
    PydanticNode,
    SetNode,
    TupleNode,
)
from furu.results.paths import (
    LogicalPath,
    MappingKeyToken,
    PathToken,
    artifact_dir_for,
    artifact_path_for,
    child_path,
    format_logical_path,
    key_path,
    mapping_key_token_for,
)
from furu.results.registry import ResultRegistry
from furu.results.rules import (
    ResultRule,
    ResolveContext,
    SavePlan,
    child_annotation_for,
    resolve_plan,
)
from furu.utils import JsonValue
from furu.utils import fully_qualified_name


@dataclass(frozen=True, slots=True)
class DumpContext:
    bundle_dir: Path
    artifacts_dir: Path
    logical_path: LogicalPath
    registry: ResultRegistry
    rules: tuple[ResultRule, ...]
    annotation: object | None = None
    current_type_id: str | None = None

    @property
    def artifact_dir(self) -> Path:
        return artifact_dir_for(self.logical_path, self.artifacts_dir)

    def child(
        self, token: PathToken, *, annotation: object | None = None
    ) -> "DumpContext":
        return replace(
            self,
            logical_path=child_path(self.logical_path, token),
            annotation=annotation,
        )

    def key_child(
        self,
        token: MappingKeyToken,
        *,
        annotation: object | None = None,
    ) -> "DumpContext":
        return replace(
            self,
            logical_path=key_path(self.logical_path, token),
            annotation=annotation,
        )

    def external(
        self,
        *,
        codec: str,
        lazy: bool = False,
        meta: dict[str, JsonValue] | None = None,
        type_id: str | None = None,
    ) -> ExternalNode:
        return ExternalNode(
            codec=codec,
            path=artifact_path_for(self.logical_path),
            lazy=lazy,
            meta=meta,
            type_id=self.current_type_id if type_id is None else type_id,
        )


@dataclass(frozen=True, slots=True)
class LoadContext:
    bundle_dir: Path
    artifacts_dir: Path
    logical_path: LogicalPath
    registry: ResultRegistry
    annotation: object | None = None

    @property
    def artifact_dir(self) -> Path:
        return artifact_dir_for(self.logical_path, self.artifacts_dir)

    def child(
        self, token: PathToken, *, annotation: object | None = None
    ) -> "LoadContext":
        return replace(
            self,
            logical_path=child_path(self.logical_path, token),
            annotation=annotation,
        )

    def key_child(
        self,
        token: MappingKeyToken,
        *,
        annotation: object | None = None,
    ) -> "LoadContext":
        return replace(
            self,
            logical_path=key_path(self.logical_path, token),
            annotation=annotation,
        )


def dump_manifest(value: object, ctx: DumpContext) -> ManifestNode:
    return _dump(value, ctx, active={})


def load_manifest(node: ManifestNode, ctx: LoadContext) -> object:
    return _load(node, ctx)


def _dump(
    value: object,
    ctx: DumpContext,
    *,
    active: dict[int, LogicalPath],
) -> ManifestNode:
    raw_value, plan = resolve_plan(
        ResolveContext(
            value=value,
            logical_path=ctx.logical_path,
            annotation=ctx.annotation,
            registry=ctx.registry,
            rules=ctx.rules,
        )
    )
    with _cycle_guard(raw_value, ctx.logical_path, active):
        if plan.mode == "inline":
            return _dump_inline(raw_value, ctx, active=active)
        if plan.mode == "external":
            return _dump_external(raw_value, plan, ctx)
        return _dump_structural(raw_value, plan, ctx, active=active)


def _dump_inline(
    value: object,
    ctx: DumpContext,
    *,
    active: dict[int, LogicalPath],
) -> ManifestNode:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [
            _dump(
                item,
                ctx.child(i, annotation=child_annotation_for(ctx.annotation, i)),
                active=active,
            )
            for i, item in enumerate(value)
        ]
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        if all(isinstance(key, str) and key != "$furu" for key in mapping):
            out: dict[str, ManifestNode] = {}
            for key, item in mapping.items():
                assert isinstance(key, str)
                out[key] = _dump(
                    item,
                    ctx.child(
                        key,
                        annotation=child_annotation_for(ctx.annotation, key),
                    ),
                    active=active,
                )
            return out
        return _dump_mapping(
            cast(Mapping[object, object], mapping),
            ctx,
            active=active,
        )
    raise TypeError(f"unexpected inline value type {type(value).__name__}")


def _dump_external(value: object, plan: SavePlan, ctx: DumpContext) -> ExternalNode:
    assert plan.codec_id is not None
    codec = ctx.registry.get_codec(plan.codec_id, logical_path=ctx.logical_path)
    artifact_dir = ctx.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    meta = codec.dump(value, artifact_dir, ctx)
    return ExternalNode(
        codec=plan.codec_id,
        path=artifact_path_for(ctx.logical_path),
        lazy=plan.lazy,
        meta=meta,
    )


def _dump_structural(
    value: object,
    plan: SavePlan,
    ctx: DumpContext,
    *,
    active: dict[int, LogicalPath],
) -> ManifestNode:
    if plan.protocol:
        protocol_ctx = replace(ctx, current_type_id=plan.type_id)
        dump_method = cast(
            Callable[[DumpContext], ManifestNode],
            getattr(value, "__furu_result_dump__"),
        )
        return dump_method(protocol_ctx)

    if is_dataclass(value):
        type_hints = _type_hints(type(value))
        fields = {
            field.name: _dump(
                getattr(value, field.name),
                ctx.child(field.name, annotation=type_hints.get(field.name)),
                active=active,
            )
            for field in dataclass_fields(value)
        }
        return DataclassNode(
            type_id=plan.type_id or fully_qualified_name(type(value)), fields=fields
        )

    if isinstance(value, PydanticBaseModel):
        type_hints = _type_hints(type(value))
        fields = {
            name: _dump(
                getattr(value, name),
                ctx.child(name, annotation=type_hints.get(name)),
                active=active,
            )
            for name in type(value).model_fields
        }
        return PydanticNode(
            type_id=plan.type_id or fully_qualified_name(type(value)),
            fields=fields,
        )

    if isinstance(value, tuple):
        return TupleNode(
            tuple(
                _dump(
                    item,
                    ctx.child(i, annotation=child_annotation_for(ctx.annotation, i)),
                    active=active,
                )
                for i, item in enumerate(value)
            )
        )

    if isinstance(value, set | frozenset):
        items = sorted(value, key=_ordering_key)
        return SetNode(
            items=tuple(
                _dump(
                    item,
                    ctx.child(i, annotation=child_annotation_for(ctx.annotation, i)),
                    active=active,
                )
                for i, item in enumerate(items)
            ),
            frozen=isinstance(value, frozenset),
        )

    if isinstance(value, Mapping):
        return _dump_mapping(
            cast(Mapping[object, object], value),
            ctx,
            active=active,
        )

    raise TypeError(f"unexpected structural value type {type(value).__name__}")


def _dump_mapping(
    value: Mapping[object, object],
    ctx: DumpContext,
    *,
    active: dict[int, LogicalPath],
) -> MappingNode:
    items: list[MappingItem] = []
    for key, item in sorted(value.items(), key=lambda pair: _ordering_key(pair[0])):
        token = mapping_key_token_for(key)
        key_node = _dump(key, ctx.key_child(token), active=active)
        value_node = _dump(
            item,
            ctx.child(token, annotation=child_annotation_for(ctx.annotation, "$value")),
            active=active,
        )
        items.append(MappingItem(key=key_node, value=value_node))
    return MappingNode(tuple(items))


def _load(node: ManifestNode, ctx: LoadContext) -> object:
    if node is None or isinstance(node, bool | int | float | str):
        return node
    if isinstance(node, list):
        return [
            _load(
                item,
                ctx.child(i, annotation=child_annotation_for(ctx.annotation, i)),
            )
            for i, item in enumerate(node)
        ]
    if isinstance(node, dict):
        return {
            key: _load(
                item,
                ctx.child(key, annotation=child_annotation_for(ctx.annotation, key)),
            )
            for key, item in node.items()
        }
    if isinstance(node, ExternalNode):
        return _load_external(node, ctx)
    if isinstance(node, DataclassNode):
        return _load_dataclass(node, ctx)
    if isinstance(node, PydanticNode):
        return _load_pydantic(node, ctx)
    if isinstance(node, TupleNode):
        return tuple(
            _load(
                item,
                ctx.child(i, annotation=child_annotation_for(ctx.annotation, i)),
            )
            for i, item in enumerate(node.items)
        )
    if isinstance(node, SetNode):
        values = [
            _load(
                item,
                ctx.child(i, annotation=child_annotation_for(ctx.annotation, i)),
            )
            for i, item in enumerate(node.items)
        ]
        return frozenset(values) if node.frozen else set(values)
    if isinstance(node, MappingNode):
        out: dict[object, object] = {}
        for item in node.items:
            key = _load(item.key, ctx)
            token = mapping_key_token_for(key)
            out[key] = _load(
                item.value,
                ctx.child(
                    token, annotation=child_annotation_for(ctx.annotation, "$value")
                ),
            )
        return out
    raise TypeError(f"unexpected manifest node type {type(node).__name__}")


def _load_external(node: ExternalNode, ctx: LoadContext) -> object:
    if node.lazy:
        return LazyValue.from_loader(
            lambda: _load_external_eager(node, ctx),
            meta=node.meta,
            codec=node.codec,
        )
    return _load_external_eager(node, ctx)


def _load_external_eager(node: ExternalNode, ctx: LoadContext) -> object:
    expected_path = artifact_path_for(ctx.logical_path)
    if node.path != expected_path:
        raise ValueError(
            "external artifact path mismatch at "
            f"{format_logical_path(ctx.logical_path)}: manifest uses {node.path!r}, "
            f"expected {expected_path!r}"
        )

    artifact_dir = ctx.bundle_dir / node.path
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"missing artifact directory {artifact_dir} at {format_logical_path(ctx.logical_path)}"
        )

    if node.type_id is not None:
        tp = _load_type(node.type_id)
        if hasattr(tp, "__furu_result_load__"):
            load_method = cast(
                Callable[[ManifestNode, LoadContext], object],
                getattr(tp, "__furu_result_load__"),
            )
            return load_method(node, ctx)

    codec = ctx.registry.get_codec(node.codec, logical_path=ctx.logical_path)
    return codec.load(artifact_dir, node.meta, ctx)


def _load_dataclass(node: DataclassNode, ctx: LoadContext) -> object:
    tp = _load_type(node.type_id)
    if not is_dataclass(tp):
        raise ValueError(f"{node.type_id!r} is not a dataclass type")

    type_hints = _type_hints(tp)
    values = {
        field.name: _load(
            node.fields[field.name],
            ctx.child(field.name, annotation=type_hints.get(field.name)),
        )
        for field in dataclass_fields(tp)
    }
    return tp(**values)


def _load_pydantic(node: PydanticNode, ctx: LoadContext) -> object:
    tp = _load_type(node.type_id)
    if not isinstance(tp, type) or not issubclass(tp, PydanticBaseModel):
        raise ValueError(f"{node.type_id!r} is not a pydantic model type")

    type_hints = _type_hints(tp)
    values = {
        name: _load(
            node.fields[name],
            ctx.child(name, annotation=type_hints.get(name)),
        )
        for name in tp.model_fields
    }
    return tp.model_construct(_fields_set=set(values), **cast(dict[str, Any], values))


@contextmanager
def _cycle_guard(
    value: object,
    logical_path: LogicalPath,
    active: dict[int, LogicalPath],
):
    if _is_scalar(value):
        yield
        return

    marker = id(value)
    if marker in active:
        raise ValueError(
            "cycle detected while saving results at "
            f"{format_logical_path(logical_path)}; the same object is already active at "
            f"{format_logical_path(active[marker])}"
        )
    active[marker] = logical_path
    try:
        yield
    finally:
        active.pop(marker, None)


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, bool | int | float | str)


def _ordering_key(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


@lru_cache(maxsize=None)
def _type_hints(tp: type[object]) -> dict[str, object]:
    return get_type_hints(tp, include_extras=True)


@lru_cache(maxsize=None)
def _load_type(type_id: str) -> type[object]:
    module_name, _, qualname = type_id.rpartition(".")
    if not module_name:
        raise ValueError(f"invalid type id {type_id!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, qualname)
    if not isinstance(value, type):
        raise ValueError(f"{type_id!r} does not resolve to a type")
    return value
