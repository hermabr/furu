from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from typing import Annotated, Literal, cast, get_args, get_origin

from pydantic import BaseModel as PydanticBaseModel

from furu.results.lazy import LazyValue
from furu.results.paths import LogicalPath, format_logical_path
from furu.results.registry import ResultCodec, ResultRegistry, ensure_codec_id
from furu.results.nodes import WRAPPER_KEY
from furu.utils import fully_qualified_name


@dataclass(frozen=True, slots=True)
class SaveWith:
    codec: str | ResultCodec | None = None
    lazy: bool | None = None


@dataclass(frozen=True, slots=True)
class ValueOverride:
    value: object
    codec: str | ResultCodec | None = None
    lazy: bool = False


@dataclass(frozen=True, slots=True)
class RuleAction:
    codec: str | ResultCodec | None = None
    lazy: bool | None = None


@dataclass(frozen=True, slots=True)
class MatchSpec:
    kind: Literal["type", "path"]
    value: object


@dataclass(frozen=True, slots=True)
class ResultRule:
    match: MatchSpec
    action: RuleAction


@dataclass(frozen=True, slots=True)
class SavePlan:
    mode: Literal["inline", "external", "structural"]
    codec_id: str | None = None
    lazy: bool = False
    type_id: str | None = None
    protocol: bool = False


@dataclass(frozen=True, slots=True)
class ResolveContext:
    value: object
    logical_path: LogicalPath
    annotation: object | None
    registry: ResultRegistry
    rules: tuple[ResultRule, ...]


@dataclass(frozen=True, slots=True)
class AnnotationPlan:
    inner_annotation: object | None
    action: RuleAction | None
    force_external: bool


@dataclass(frozen=True, slots=True)
class ResultRuleBuilder:
    match: MatchSpec

    def save_as(self, codec: str | ResultCodec) -> ResultRule:
        return ResultRule(match=self.match, action=RuleAction(codec=codec, lazy=False))

    def lazy(self, *, codec: str | ResultCodec | None = None) -> ResultRule:
        return ResultRule(match=self.match, action=RuleAction(codec=codec, lazy=True))


def when_type(tp: type[object]) -> ResultRuleBuilder:
    return ResultRuleBuilder(match=MatchSpec(kind="type", value=tp))


def at(*path: str | int) -> ResultRuleBuilder:
    return ResultRuleBuilder(match=MatchSpec(kind="path", value=tuple(path)))


def resolve_plan(ctx: ResolveContext) -> tuple[object, SavePlan]:
    value = ctx.value

    if isinstance(value, ValueOverride):
        raw_value = value.value
        return raw_value, _external_plan(
            raw_value,
            ctx,
            action=RuleAction(codec=value.codec, lazy=value.lazy),
        )

    if isinstance(value, LazyValue):
        raw_value = value.value_for_save()
        return raw_value, _external_plan(
            raw_value,
            ctx,
            action=RuleAction(
                codec=cast(str | ResultCodec | None, value.codec),
                lazy=True,
            ),
        )

    path_rule = _matching_rule(ctx.rules, kind="path", value=ctx.logical_path)
    if path_rule is not None:
        return value, _external_plan(value, ctx, action=path_rule.action)

    annotation_plan = annotation_plan_for(ctx.annotation)
    if annotation_plan.action is not None:
        return value, _external_plan(value, ctx, action=annotation_plan.action)

    if _supports_result_protocol(value):
        return value, SavePlan(
            mode="structural",
            type_id=fully_qualified_name(type(value)),
            protocol=True,
        )

    default_codec_id = ctx.registry.resolve_default_codec_id(value)
    if default_codec_id is not None:
        return value, SavePlan(mode="external", codec_id=default_codec_id)

    if _is_inline_candidate(value):
        return value, SavePlan(mode="inline")

    if is_dataclass(value):
        return value, SavePlan(
            mode="structural",
            type_id=fully_qualified_name(type(value)),
        )
    if isinstance(value, PydanticBaseModel):
        return value, SavePlan(
            mode="structural",
            type_id=fully_qualified_name(type(value)),
        )
    if isinstance(value, tuple | set | frozenset | Mapping):
        return value, SavePlan(mode="structural")

    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    raise TypeError(
        "unsupported result value at "
        f"{format_logical_path(ctx.logical_path)} for type {type_name}. "
        "Register a codec, wrap the value with save_with(...), add a path/type rule, "
        "or implement the Furu result protocol."
    )


def annotation_plan_for(annotation: object | None) -> AnnotationPlan:
    if annotation is None:
        return AnnotationPlan(inner_annotation=None, action=None, force_external=False)

    base = annotation
    metadata: tuple[object, ...] = ()
    origin = get_origin(base)
    if origin is Annotated:
        args = get_args(base)
        base = args[0]
        metadata = tuple(args[1:])

    lazy = False
    lazy_origin = get_origin(base)
    if lazy_origin is LazyValue:
        args = get_args(base)
        base = args[0] if args else object
        lazy = True

    save_with = next(
        (item for item in reversed(metadata) if isinstance(item, SaveWith)),
        None,
    )
    if save_with is None and not lazy:
        return AnnotationPlan(inner_annotation=base, action=None, force_external=False)

    action = RuleAction(
        codec=None if save_with is None else save_with.codec,
        lazy=lazy if save_with is None or save_with.lazy is None else save_with.lazy,
    )
    return AnnotationPlan(inner_annotation=base, action=action, force_external=True)


def child_annotation_for(annotation: object | None, token: object) -> object | None:
    plan = annotation_plan_for(annotation)
    base = plan.inner_annotation
    if base is None:
        return None

    origin = get_origin(base)
    args = get_args(base)

    if origin in (list, set, frozenset):
        return args[0] if args else None

    if origin is tuple:
        if not args:
            return None
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        if isinstance(token, int) and token < len(args):
            return args[token]
        return None

    if origin in (dict, Mapping):
        if len(args) != 2:
            return None
        if token == WRAPPER_KEY:
            return None
        return args[1]

    return None


def _external_plan(
    value: object,
    ctx: ResolveContext,
    *,
    action: RuleAction,
) -> SavePlan:
    codec_id = None
    if action.codec is not None:
        codec_id = ensure_codec_id(action.codec, ctx.registry)
    elif (default_codec_id := ctx.registry.resolve_default_codec_id(value)) is not None:
        codec_id = default_codec_id
    elif _is_json_file_candidate(value):
        codec_id = "json.file.v1"

    if codec_id is None:
        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        raise TypeError(
            "no result codec resolved at "
            f"{format_logical_path(ctx.logical_path)} for type {type_name}. "
            "Register a codec, choose one with save_with(...), add a path/type rule, "
            "or implement the Furu result protocol."
        )

    return SavePlan(mode="external", codec_id=codec_id, lazy=bool(action.lazy))


def _matching_rule(
    rules: tuple[ResultRule, ...],
    *,
    kind: Literal["type", "path"],
    value: object,
) -> ResultRule | None:
    matched: ResultRule | None = None
    for rule in rules:
        if rule.match.kind != kind:
            continue
        if kind == "path":
            if rule.match.value == value:
                matched = rule
        else:
            tp = rule.match.value
            if isinstance(tp, type) and isinstance(value, tp):
                matched = rule
    return matched


def _supports_result_protocol(value: object) -> bool:
    return hasattr(value, "__furu_result_dump__") and hasattr(
        type(value), "__furu_result_load__"
    )


def _is_inline_candidate(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list):
        return True
    return isinstance(value, dict)


def _is_json_file_candidate(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list):
        return all(_is_json_file_candidate(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str)
            and key != WRAPPER_KEY
            and _is_json_file_candidate(item)
            for key, item in value.items()
        )
    return False
