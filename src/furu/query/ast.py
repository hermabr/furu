"""AST models for experiment query filtering."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


Scalar: TypeAlias = str | int | float | bool | None


class _QueryNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrueQuery(_QueryNode):
    op: Literal["true"] = "true"


class FalseQuery(_QueryNode):
    op: Literal["false"] = "false"


class AndQuery(_QueryNode):
    op: Literal["and"] = "and"
    args: list[Query] = Field(min_length=1)


class OrQuery(_QueryNode):
    op: Literal["or"] = "or"
    args: list[Query] = Field(min_length=1)


class NotQuery(_QueryNode):
    op: Literal["not"] = "not"
    arg: Query


class ExistsQuery(_QueryNode):
    op: Literal["exists"] = "exists"
    path: str


class MissingQuery(_QueryNode):
    op: Literal["missing"] = "missing"
    path: str


class EqQuery(_QueryNode):
    op: Literal["eq"] = "eq"
    path: str
    value: Scalar


class NeQuery(_QueryNode):
    op: Literal["ne"] = "ne"
    path: str
    value: Scalar


class LtQuery(_QueryNode):
    op: Literal["lt"] = "lt"
    path: str
    value: Scalar


class LteQuery(_QueryNode):
    op: Literal["lte"] = "lte"
    path: str
    value: Scalar


class GtQuery(_QueryNode):
    op: Literal["gt"] = "gt"
    path: str
    value: Scalar


class GteQuery(_QueryNode):
    op: Literal["gte"] = "gte"
    path: str
    value: Scalar


class BetweenQuery(_QueryNode):
    op: Literal["between"] = "between"
    path: str
    low: Scalar
    high: Scalar
    inclusive: Literal["both", "left", "right", "none"] = "both"


class InQuery(_QueryNode):
    op: Literal["in"] = "in"
    path: str
    values: list[Scalar]


class NinQuery(_QueryNode):
    op: Literal["nin"] = "nin"
    path: str
    values: list[Scalar]


class ContainsQuery(_QueryNode):
    op: Literal["contains"] = "contains"
    path: str
    value: Scalar
    case_sensitive: bool


class StartsWithQuery(_QueryNode):
    op: Literal["startswith"] = "startswith"
    path: str
    prefix: str
    case_sensitive: bool


class EndsWithQuery(_QueryNode):
    op: Literal["endswith"] = "endswith"
    path: str
    suffix: str
    case_sensitive: bool


class RegexQuery(_QueryNode):
    op: Literal["regex"] = "regex"
    path: str
    pattern: str
    flags: str


class TypeIsQuery(_QueryNode):
    op: Literal["type_is"] = "type_is"
    path: str
    type: str


class IsAQuery(_QueryNode):
    op: Literal["is_a"] = "is_a"
    path: str
    type: str


class RelatedToQuery(_QueryNode):
    op: Literal["related_to"] = "related_to"
    path: str
    type: str


Query: TypeAlias = Annotated[
    TrueQuery
    | FalseQuery
    | AndQuery
    | OrQuery
    | NotQuery
    | ExistsQuery
    | MissingQuery
    | EqQuery
    | NeQuery
    | LtQuery
    | LteQuery
    | GtQuery
    | GteQuery
    | BetweenQuery
    | InQuery
    | NinQuery
    | ContainsQuery
    | StartsWithQuery
    | EndsWithQuery
    | RegexQuery
    | TypeIsQuery
    | IsAQuery
    | RelatedToQuery,
    Field(discriminator="op"),
]


__all__ = [
    "Query",
    "Scalar",
    "AndQuery",
    "ContainsQuery",
    "EndsWithQuery",
    "EqQuery",
    "ExistsQuery",
    "FalseQuery",
    "GtQuery",
    "GteQuery",
    "InQuery",
    "IsAQuery",
    "LteQuery",
    "LtQuery",
    "MissingQuery",
    "NeQuery",
    "NinQuery",
    "NotQuery",
    "OrQuery",
    "BetweenQuery",
    "RegexQuery",
    "RelatedToQuery",
    "StartsWithQuery",
    "TrueQuery",
    "TypeIsQuery",
]
