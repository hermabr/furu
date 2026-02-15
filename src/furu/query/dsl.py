from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .ast import (
    AndNode,
    BetweenNode,
    ContainsNode,
    EndswithNode,
    EqNode,
    ExistsNode,
    FalseNode,
    GtNode,
    GteNode,
    InNode,
    IsANode,
    LtNode,
    LteNode,
    MissingNode,
    NeNode,
    NinNode,
    NotNode,
    OrNode,
    Query,
    RegexNode,
    RelatedToNode,
    Scalar,
    StartswithNode,
    TrueNode,
    TypeIsNode,
)


class _HasClass(Protocol):
    @property
    def __class__(self) -> type: ...


def _type_to_string(value: str | type | _HasClass) -> str:
    if isinstance(value, str):
        return value
    cls = value if isinstance(value, type) else value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _flatten_and_args(left: Query, right: Query) -> list[Query]:
    args: list[Query] = []
    if isinstance(left, AndNode):
        args.extend(left.args)
    else:
        args.append(left)
    if isinstance(right, AndNode):
        args.extend(right.args)
    else:
        args.append(right)
    return args


def _flatten_or_args(left: Query, right: Query) -> list[Query]:
    args: list[Query] = []
    if isinstance(left, OrNode):
        args.extend(left.args)
    else:
        args.append(left)
    if isinstance(right, OrNode):
        args.extend(right.args)
    else:
        args.append(right)
    return args


def _normalize_values(values: tuple[Scalar, ...]) -> list[Scalar]:
    return [*values]


@dataclass(frozen=True)
class QueryExpr:
    node: Query

    def __and__(self, other: QueryExpr) -> QueryExpr:
        return QueryExpr(AndNode(args=_flatten_and_args(self.node, other.node)))

    def __or__(self, other: QueryExpr) -> QueryExpr:
        return QueryExpr(OrNode(args=_flatten_or_args(self.node, other.node)))

    def __invert__(self) -> QueryExpr:
        return QueryExpr(NotNode(arg=self.node))

    def __bool__(self) -> bool:
        raise TypeError(
            "Query expressions cannot be used as booleans; use '&' and '|' operators"
        )

    def to_ast(self) -> Query:
        return self.node


@dataclass(frozen=True)
class FieldRef:
    path: str

    def __getattr__(self, segment: str) -> FieldRef:
        if segment.startswith("_"):
            raise AttributeError(segment)
        return FieldRef(path=f"{self.path}.{segment}")

    def __getitem__(self, key: int | str) -> FieldRef:
        if isinstance(key, int):
            if key < 0:
                raise ValueError("Negative indices are not supported in query paths")
            return FieldRef(path=f"{self.path}.{key}")
        if not key:
            raise ValueError("String keys in query paths cannot be empty")
        if "." in key:
            raise ValueError("String keys in query paths cannot contain '.'")
        return FieldRef(path=f"{self.path}.{key}")

    def __eq__(self, value: Scalar) -> QueryExpr:  # type: ignore[override]
        return QueryExpr(EqNode(path=self.path, value=value))

    def __ne__(self, value: Scalar) -> QueryExpr:  # type: ignore[override]
        return QueryExpr(NeNode(path=self.path, value=value))

    def __lt__(self, value: Scalar) -> QueryExpr:
        return QueryExpr(LtNode(path=self.path, value=value))

    def __le__(self, value: Scalar) -> QueryExpr:
        return QueryExpr(LteNode(path=self.path, value=value))

    def __gt__(self, value: Scalar) -> QueryExpr:
        return QueryExpr(GtNode(path=self.path, value=value))

    def __ge__(self, value: Scalar) -> QueryExpr:
        return QueryExpr(GteNode(path=self.path, value=value))

    def exists(self) -> QueryExpr:
        return QueryExpr(ExistsNode(path=self.path))

    def missing(self) -> QueryExpr:
        return QueryExpr(MissingNode(path=self.path))

    def between(
        self,
        low: Scalar,
        high: Scalar,
        inclusive: Literal["both", "left", "right", "none"] = "both",
    ) -> QueryExpr:
        return QueryExpr(
            BetweenNode(path=self.path, low=low, high=high, inclusive=inclusive)
        )

    def in_(
        self,
        *values: Scalar,
    ) -> QueryExpr:
        return QueryExpr(InNode(path=self.path, values=_normalize_values(values)))

    def not_in(
        self,
        *values: Scalar,
    ) -> QueryExpr:
        return QueryExpr(NinNode(path=self.path, values=_normalize_values(values)))

    def contains(self, value: Scalar, case_sensitive: bool = True) -> QueryExpr:
        return QueryExpr(
            ContainsNode(path=self.path, value=value, case_sensitive=case_sensitive)
        )

    def startswith(self, prefix: str, case_sensitive: bool = True) -> QueryExpr:
        return QueryExpr(
            StartswithNode(path=self.path, prefix=prefix, case_sensitive=case_sensitive)
        )

    def endswith(self, suffix: str, case_sensitive: bool = True) -> QueryExpr:
        return QueryExpr(
            EndswithNode(path=self.path, suffix=suffix, case_sensitive=case_sensitive)
        )

    def regex(self, pattern: str, flags: str = "") -> QueryExpr:
        return QueryExpr(RegexNode(path=self.path, pattern=pattern, flags=flags))

    def type_is(self, value: str | type | _HasClass) -> QueryExpr:
        return QueryExpr(TypeIsNode(path=self.path, type=_type_to_string(value)))

    def is_a(self, value: str | type | _HasClass) -> QueryExpr:
        return QueryExpr(IsANode(path=self.path, type=_type_to_string(value)))

    def related_to(self, value: str | type | _HasClass) -> QueryExpr:
        return QueryExpr(RelatedToNode(path=self.path, type=_type_to_string(value)))


@dataclass(frozen=True)
class _QRoot:
    exp: FieldRef = FieldRef("exp")
    config: FieldRef = FieldRef("config")
    meta: FieldRef = FieldRef("meta")
    state: FieldRef = FieldRef("state")


Q = _QRoot()
TRUE = QueryExpr(TrueNode())
FALSE = QueryExpr(FalseNode())
