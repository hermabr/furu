from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Scalar: TypeAlias = str | int | float | bool | None


class _QueryNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrueNode(_QueryNode):
    op: Literal["true"] = "true"


class FalseNode(_QueryNode):
    op: Literal["false"] = "false"


class AndNode(_QueryNode):
    op: Literal["and"] = "and"
    args: list[Query] = Field(min_length=1)


class OrNode(_QueryNode):
    op: Literal["or"] = "or"
    args: list[Query] = Field(min_length=1)


class NotNode(_QueryNode):
    op: Literal["not"] = "not"
    arg: Query


class ExistsNode(_QueryNode):
    op: Literal["exists"] = "exists"
    path: str


class MissingNode(_QueryNode):
    op: Literal["missing"] = "missing"
    path: str


class EqNode(_QueryNode):
    op: Literal["eq"] = "eq"
    path: str
    value: Scalar


class NeNode(_QueryNode):
    op: Literal["ne"] = "ne"
    path: str
    value: Scalar


class LtNode(_QueryNode):
    op: Literal["lt"] = "lt"
    path: str
    value: Scalar


class LteNode(_QueryNode):
    op: Literal["lte"] = "lte"
    path: str
    value: Scalar


class GtNode(_QueryNode):
    op: Literal["gt"] = "gt"
    path: str
    value: Scalar


class GteNode(_QueryNode):
    op: Literal["gte"] = "gte"
    path: str
    value: Scalar


class BetweenNode(_QueryNode):
    op: Literal["between"] = "between"
    path: str
    low: Scalar
    high: Scalar
    inclusive: Literal["both", "left", "right", "none"]


class InNode(_QueryNode):
    op: Literal["in"] = "in"
    path: str
    values: list[Scalar] = Field(min_length=1)


class NinNode(_QueryNode):
    op: Literal["nin"] = "nin"
    path: str
    values: list[Scalar] = Field(min_length=1)


class ContainsNode(_QueryNode):
    op: Literal["contains"] = "contains"
    path: str
    value: Scalar
    case_sensitive: bool


class StartswithNode(_QueryNode):
    op: Literal["startswith"] = "startswith"
    path: str
    prefix: str
    case_sensitive: bool


class EndswithNode(_QueryNode):
    op: Literal["endswith"] = "endswith"
    path: str
    suffix: str
    case_sensitive: bool


class RegexNode(_QueryNode):
    op: Literal["regex"] = "regex"
    path: str
    pattern: str
    flags: str


class TypeIsNode(_QueryNode):
    op: Literal["type_is"] = "type_is"
    path: str
    type: str


class IsANode(_QueryNode):
    op: Literal["is_a"] = "is_a"
    path: str
    type: str


class RelatedToNode(_QueryNode):
    op: Literal["related_to"] = "related_to"
    path: str
    type: str


Query: TypeAlias = Annotated[
    TrueNode
    | FalseNode
    | AndNode
    | OrNode
    | NotNode
    | ExistsNode
    | MissingNode
    | EqNode
    | NeNode
    | LtNode
    | LteNode
    | GtNode
    | GteNode
    | BetweenNode
    | InNode
    | NinNode
    | ContainsNode
    | StartswithNode
    | EndswithNode
    | RegexNode
    | TypeIsNode
    | IsANode
    | RelatedToNode,
    Field(discriminator="op"),
]


AndNode.model_rebuild()
OrNode.model_rebuild()
NotNode.model_rebuild()
