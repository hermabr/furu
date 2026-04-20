from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Self,
    TypeVar,
    get_args,
    get_origin,
)

from furu.results.paths import LogicalPath, PathToken

if TYPE_CHECKING:
    from furu.results.codecs import DumpContext, LoadContext, ResultCodec
    from furu.results.registry import ResultRegistry

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResultValue(Generic[T]):
    value: T
    codec: str | ResultCodec[Any] | None = None
    lazy: bool | None = None


@dataclass(frozen=True, slots=True)
class ResultPolicy:
    codec: str | ResultCodec[Any] | None = None
    lazy: bool | None = None

    def merged_with(self, other: "ResultPolicy") -> "ResultPolicy":
        return ResultPolicy(
            codec=self.codec if other.codec is None else other.codec,
            lazy=self.lazy if other.lazy is None else other.lazy,
        )


def merge_policies(*policies: ResultPolicy) -> ResultPolicy:
    merged = ResultPolicy()
    for policy in policies:
        merged = merged.merged_with(policy)
    return merged


def result(
    value: T,
    *,
    codec: str | ResultCodec[Any] | None = None,
    lazy: bool | None = None,
) -> ResultValue[T]:
    return ResultValue(value=value, codec=codec, lazy=lazy)


def lazy(
    value: T,
    *,
    codec: str | ResultCodec[Any] | None = None,
) -> ResultValue[T]:
    return result(value, codec=codec, lazy=True)


def unwrap_result_value(value: T | ResultValue[T]) -> tuple[ResultPolicy, T]:
    if isinstance(value, ResultValue):
        return ResultPolicy(codec=value.codec, lazy=value.lazy), value.value
    return ResultPolicy(), value


@dataclass(frozen=True, slots=True)
class SaveWith:
    codec: str | ResultCodec[Any] | None = None
    lazy: bool | None = None

    def to_policy(self) -> ResultPolicy:
        return ResultPolicy(codec=self.codec, lazy=self.lazy)


class FuruResult:
    def __furu_result_dump__(self, ctx: "DumpContext") -> Any:
        raise NotImplementedError

    @classmethod
    def __furu_result_load__(cls, ctx: "LoadContext", meta: Any) -> Any:
        raise NotImplementedError


def supports_furu_result_protocol(value: object) -> bool:
    dump = getattr(value, "__furu_result_dump__", None)
    load = getattr(type(value), "__furu_result_load__", None)
    return callable(dump) and callable(load)


@dataclass(frozen=True, slots=True)
class _BaseRule:
    policy: ResultPolicy = field(default_factory=ResultPolicy)

    def save_with(self, codec: str | ResultCodec[Any]) -> Self:
        return replace(self, policy=replace(self.policy, codec=codec))

    def lazy(self, value: bool = True) -> Self:
        return replace(self, policy=replace(self.policy, lazy=value))


@dataclass(frozen=True, slots=True)
class ResultAtRule(_BaseRule):
    segments: tuple[PathToken, ...] = ()

    def matches(self, logical_path: LogicalPath, _: object) -> bool:
        return logical_path.segments == self.segments


@dataclass(frozen=True, slots=True)
class ResultWhenTypeRule(_BaseRule):
    tp: type[object] | tuple[type[object], ...] = object

    def matches(self, _: LogicalPath, value: object) -> bool:
        return isinstance(value, self.tp)


type ResultRule = ResultAtRule | ResultWhenTypeRule


def result_at(*segments: PathToken) -> ResultAtRule:
    return ResultAtRule(segments=tuple(segments))


def result_when_type(
    tp: type[object] | tuple[type[object], ...],
) -> ResultWhenTypeRule:
    return ResultWhenTypeRule(tp=tp)


def extract_save_with(annotation: object) -> ResultPolicy:
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        metadata = args[1:]
        policy = ResultPolicy()
        for item in metadata:
            if isinstance(item, SaveWith):
                policy = merge_policies(policy, item.to_policy())
        return policy
    return ResultPolicy()


@dataclass(slots=True)
class ResultConfig:
    registry: "ResultRegistry"
    rules: list[ResultRule] = field(default_factory=list)

    @classmethod
    def default(cls) -> "ResultConfig":
        from furu.results.registry import ResultRegistry

        return cls(registry=ResultRegistry.default())

    def matching_policy(self, logical_path: LogicalPath, value: object) -> ResultPolicy:
        policy = ResultPolicy()
        for rule in self.rules:
            if rule.matches(logical_path, value):
                policy = merge_policies(policy, rule.policy)
        return policy
