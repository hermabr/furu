from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Annotated, Protocol, get_args, get_origin

from furu.results.paths import LogicalPath, PathToken

if TYPE_CHECKING:
    from furu.results.registry import ResultCodec


@dataclass(frozen=True, slots=True)
class SaveSpec:
    serializer: str | "ResultCodec[Any]" | None = None
    lazy: bool | None = None


@dataclass(frozen=True, slots=True)
class _WrappedValue[T]:
    value: T
    spec: SaveSpec


@dataclass(frozen=True, slots=True)
class SaveWith:
    serializer: str | "ResultCodec[Any]" | None = None
    lazy: bool | None = None

    def to_spec(self) -> SaveSpec:
        return SaveSpec(serializer=self.serializer, lazy=self.lazy)


@dataclass(frozen=True, slots=True)
class PathRule:
    path: tuple[PathToken, ...]
    spec: SaveSpec


@dataclass(frozen=True, slots=True)
class TypeRule:
    tp: type[object]
    spec: SaveSpec


type ResultRule = PathRule | TypeRule


class HasResultRules(Protocol):
    rules: tuple[ResultRule, ...]


@dataclass(frozen=True, slots=True)
class _PathRuleBuilder:
    path: tuple[PathToken, ...]

    def save_as(self, serializer: str | "ResultCodec[Any]") -> PathRule:
        return PathRule(path=self.path, spec=SaveSpec(serializer=serializer))

    def lazy(self, serializer: str | "ResultCodec[Any]" | None = None) -> PathRule:
        return PathRule(path=self.path, spec=SaveSpec(serializer=serializer, lazy=True))


@dataclass(frozen=True, slots=True)
class _TypeRuleBuilder:
    tp: type[object]

    def save_as(self, serializer: str | "ResultCodec[Any]") -> TypeRule:
        return TypeRule(tp=self.tp, spec=SaveSpec(serializer=serializer))

    def lazy(self, serializer: str | "ResultCodec[Any]" | None = None) -> TypeRule:
        return TypeRule(tp=self.tp, spec=SaveSpec(serializer=serializer, lazy=True))


def at(*path: PathToken) -> _PathRuleBuilder:
    return _PathRuleBuilder(path=path)


def when_type(tp: type[object]) -> _TypeRuleBuilder:
    return _TypeRuleBuilder(tp=tp)


def merge_specs(*specs: SaveSpec | None) -> SaveSpec:
    serializer: str | ResultCodec[Any] | None = None
    lazy: bool | None = None
    for spec in specs:
        if spec is None:
            continue
        if spec.serializer is not None:
            serializer = spec.serializer
        if spec.lazy is not None:
            lazy = spec.lazy
    return SaveSpec(serializer=serializer, lazy=lazy)


def annotation_save_spec(annotation: object | None) -> SaveSpec:
    if annotation is None or get_origin(annotation) is not Annotated:
        return SaveSpec()

    spec = SaveSpec()
    for metadata in get_args(annotation)[1:]:
        if isinstance(metadata, SaveWith):
            spec = merge_specs(spec, metadata.to_spec())
    return spec


def path_rule_spec(config: HasResultRules, path: LogicalPath) -> SaveSpec:
    spec = SaveSpec()
    for rule in config.rules:
        if isinstance(rule, PathRule) and rule.path == path.parts:
            spec = merge_specs(spec, rule.spec)
    return spec


def type_rule_spec(config: HasResultRules, value: object) -> SaveSpec:
    spec = SaveSpec()
    for rule in config.rules:
        if isinstance(rule, TypeRule) and isinstance(value, rule.tp):
            spec = merge_specs(spec, rule.spec)
    return spec
