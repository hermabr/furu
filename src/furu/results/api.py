from __future__ import annotations

from dataclasses import dataclass, field

from furu.results.lazy import LazyValue
from furu.results.registry import ResultCodec, ResultRegistry, default_result_registry
from furu.results.rules import (
    ResultRule,
    SaveWith,
    ValueOverride,
    at,
    when_type,
)


@dataclass(slots=True)
class ResultConfig:
    registry: ResultRegistry
    rules: list[ResultRule] = field(default_factory=list)


def save_with(
    value: object,
    *,
    codec: str | ResultCodec | None = None,
) -> ValueOverride:
    return ValueOverride(value=value, codec=codec)


def lazy(
    value: object,
    *,
    codec: str | ResultCodec | None = None,
) -> LazyValue[object]:
    return LazyValue.from_value(value, codec=codec)


__all__ = [
    "ResultConfig",
    "SaveWith",
    "at",
    "default_result_registry",
    "lazy",
    "save_with",
    "when_type",
]
