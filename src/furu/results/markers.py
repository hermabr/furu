from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from .paths import LogicalPath, PathToken
from .protocol import ResultCodec

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResultSpec:
    codec: str | ResultCodec[Any] | None = None
    lazy: bool | None = None


@dataclass(frozen=True, slots=True)
class FuruResult(Generic[T]):
    value: T
    codec: str | ResultCodec[Any] | None = None
    lazy: bool | None = None

    @property
    def spec(self) -> ResultSpec:
        return ResultSpec(codec=self.codec, lazy=self.lazy)


@dataclass(frozen=True, slots=True)
class SaveWith:
    codec: str | ResultCodec[Any] | None = None
    lazy: bool | None = None

    @property
    def spec(self) -> ResultSpec:
        return ResultSpec(codec=self.codec, lazy=self.lazy)


@dataclass(frozen=True, slots=True)
class ResultRule:
    path: tuple[PathToken, ...]
    spec: ResultSpec = ResultSpec()

    def matches(self, logical_path: LogicalPath) -> bool:
        return logical_path.tokens == self.path

    def save_as(self, codec: str | ResultCodec[Any]) -> "ResultRule":
        spec = merge_specs(self.spec, ResultSpec(codec=codec)) or ResultSpec()
        return ResultRule(self.path, spec)

    def lazy(self, lazy: bool = True) -> "ResultRule":
        spec = merge_specs(self.spec, ResultSpec(lazy=lazy)) or ResultSpec()
        return ResultRule(self.path, spec)


def merge_specs(*specs: ResultSpec | None) -> ResultSpec | None:
    merged = ResultSpec()
    seen = False
    for spec in specs:
        if spec is None:
            continue
        seen = True
        if spec.codec is not None:
            merged = ResultSpec(codec=spec.codec, lazy=merged.lazy)
        if spec.lazy is not None:
            merged = ResultSpec(codec=merged.codec, lazy=spec.lazy)
    return merged if seen else None


def unwrap_furu_result(value: T) -> tuple[ResultSpec | None, T]:
    if isinstance(value, FuruResult):
        return value.spec, cast(T, value.value)
    return None, value


def result(
    value: T,
    *,
    codec: str | ResultCodec[Any] | None = None,
    lazy: bool | None = None,
) -> FuruResult[T]:
    return FuruResult(value=value, codec=codec, lazy=lazy)


def save_with(
    value: T,
    *,
    codec: str | ResultCodec[Any] | None = None,
    lazy: bool | None = None,
) -> FuruResult[T]:
    return result(value, codec=codec, lazy=lazy)


def lazy(
    value: T,
    *,
    codec: str | ResultCodec[Any] | None = None,
) -> FuruResult[T]:
    return result(value, codec=codec, lazy=True)


def at(*tokens: PathToken) -> ResultRule:
    for token in tokens:
        if not isinstance(token, str | int):
            raise TypeError("path rules only support string and integer tokens")
    return ResultRule(path=tokens)
