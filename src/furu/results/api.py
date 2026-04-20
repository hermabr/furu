from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeGuard, cast

from furu.results.lazy import LazyValue
from furu.results.paths import LogicalPath
from furu.results.registry import ResultCodec, ResultRegistry


@dataclass(frozen=True)
class SaveWith:
    codec: str | ResultCodec | None = None


@dataclass(frozen=True)
class ResultRule:
    match_kind: str
    path: LogicalPath | None = None
    value_type: type[Any] | None = None
    codec: str | ResultCodec | None = None
    lazy: bool | None = None


class _PathRuleBuilder:
    def __init__(self, path: LogicalPath) -> None:
        self._path = path

    def save_as(self, codec: str | ResultCodec) -> ResultRule:
        return ResultRule(match_kind="path", path=self._path, codec=codec)

    def lazy(self, *, codec: str | ResultCodec | None = None) -> ResultRule:
        return ResultRule(match_kind="path", path=self._path, codec=codec, lazy=True)


class _TypeRuleBuilder:
    def __init__(self, value_type: type[Any]) -> None:
        self._value_type = value_type

    def save_as(self, codec: str | ResultCodec) -> ResultRule:
        return ResultRule(match_kind="type", value_type=self._value_type, codec=codec)

    def lazy(self, *, codec: str | ResultCodec | None = None) -> ResultRule:
        return ResultRule(
            match_kind="type",
            value_type=self._value_type,
            codec=codec,
            lazy=True,
        )


@dataclass(frozen=True)
class ResultConfig:
    registry: ResultRegistry = field(default_factory=ResultRegistry.default)
    rules: tuple[ResultRule, ...] = ()

    def __post_init__(self) -> None:
        registry = self.registry
        for rule in self.rules:
            codec = rule.codec
            if _is_codec_instance(codec):
                registry = registry.with_codec(codec)
        object.__setattr__(self, "registry", registry)


@dataclass(frozen=True)
class _ValueDirective:
    value: Any
    codec: str | ResultCodec | None = None
    lazy: bool | None = None


def save_with[T](
    value: T,
    *,
    codec: str | ResultCodec | None = None,
    lazy: bool | None = None,
) -> T:
    return cast(T, _ValueDirective(value=value, codec=codec, lazy=lazy))


def lazy[T](
    value: T,
    *,
    codec: str | ResultCodec | None = None,
) -> LazyValue[T]:
    return LazyValue.from_value(value, codec=codec)


def at(*segments: str | int) -> _PathRuleBuilder:
    return _PathRuleBuilder(tuple(segments))


def when_type(value_type: type[Any]) -> _TypeRuleBuilder:
    return _TypeRuleBuilder(value_type)


def unwrap_value_directive(value: Any) -> _ValueDirective | None:
    if isinstance(value, _ValueDirective):
        return value
    return None


def _is_codec_instance(value: object) -> TypeGuard[ResultCodec]:
    return (
        value is not None
        and not isinstance(value, str)
        and hasattr(value, "codec_id")
        and callable(getattr(value, "dump", None))
        and callable(getattr(value, "load", None))
    )
