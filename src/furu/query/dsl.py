"""Python DSL for building query AST nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .ast import (
    BetweenQuery,
    ContainsQuery,
    EqQuery,
    ExistsQuery,
    FalseQuery,
    GtQuery,
    GteQuery,
    InQuery,
    IsAQuery,
    LtQuery,
    LteQuery,
    MissingQuery,
    NeQuery,
    NinQuery,
    Query,
    RegexQuery,
    RelatedToQuery,
    Scalar,
    StartsWithQuery,
    TrueQuery,
    TypeIsQuery,
    EndsWithQuery,
)


@dataclass(frozen=True)
class FieldRef:
    """Reference to a dot-delimited document path."""

    path: str

    def __getattr__(self, item: str) -> "FieldRef":
        return FieldRef(f"{self.path}.{item}")

    def __getitem__(self, key: str | int) -> "FieldRef":
        if isinstance(key, int):
            if key < 0:
                raise ValueError(
                    "query index access only supports non-negative integers"
                )
            return FieldRef(f"{self.path}.{key}")

        if "." in key:
            raise ValueError("string path keys may not contain '.' in v1")

        return FieldRef(f"{self.path}.{key}")

    def _scalar(self, value: object) -> Scalar:
        if value is None:
            return None
        if isinstance(value, str | int | float | bool):
            return value
        raise TypeError(f"query value must be a scalar, got {type(value).__name__!r}")

    def __eq__(self, value: object) -> Query:  # type: ignore[override]
        return EqQuery(op="eq", path=self.path, value=self._scalar(value))

    def __ne__(self, value: object) -> Query:  # type: ignore[override]
        return NeQuery(op="ne", path=self.path, value=self._scalar(value))

    def __lt__(self, value: object) -> Query:
        return LtQuery(op="lt", path=self.path, value=self._scalar(value))

    def __le__(self, value: object) -> Query:
        return LteQuery(op="lte", path=self.path, value=self._scalar(value))

    def __gt__(self, value: object) -> Query:
        return GtQuery(op="gt", path=self.path, value=self._scalar(value))

    def __ge__(self, value: object) -> Query:
        return GteQuery(op="gte", path=self.path, value=self._scalar(value))

    def exists(self) -> Query:
        return ExistsQuery(op="exists", path=self.path)

    def missing(self) -> Query:
        return MissingQuery(op="missing", path=self.path)

    def between(
        self,
        low: Scalar,
        high: Scalar,
        inclusive: Literal["both", "left", "right", "none"] = "both",
    ) -> Query:
        return BetweenQuery(
            op="between",
            path=self.path,
            low=self._scalar(low),
            high=self._scalar(high),
            inclusive=inclusive,
        )

    def in_(self, *values: Scalar) -> Query:
        return InQuery(
            op="in", path=self.path, values=[self._scalar(value) for value in values]
        )

    def not_in(self, *values: Scalar) -> Query:
        return NinQuery(
            op="nin", path=self.path, values=[self._scalar(value) for value in values]
        )

    def contains(self, value: Scalar, *, case_sensitive: bool = True) -> Query:
        if not isinstance(value, str):
            raise TypeError(
                f"contains value must be a string, got {type(value).__name__!r}"
            )
        return ContainsQuery(
            op="contains",
            path=self.path,
            value=value,
            case_sensitive=case_sensitive,
        )

    def startswith(self, prefix: str, *, case_sensitive: bool = True) -> Query:
        if not isinstance(prefix, str):
            raise TypeError(
                f"startswith prefix must be a string, got {type(prefix).__name__!r}"
            )
        return StartsWithQuery(
            op="startswith",
            path=self.path,
            prefix=prefix,
            case_sensitive=case_sensitive,
        )

    def endswith(self, suffix: str, *, case_sensitive: bool = True) -> Query:
        if not isinstance(suffix, str):
            raise TypeError(
                f"endswith suffix must be a string, got {type(suffix).__name__!r}"
            )
        return EndsWithQuery(
            op="endswith",
            path=self.path,
            suffix=suffix,
            case_sensitive=case_sensitive,
        )

    def regex(self, pattern: str, *, flags: str = "") -> Query:
        if not isinstance(pattern, str):
            raise TypeError(
                f"regex pattern must be a string, got {type(pattern).__name__!r}"
            )
        return RegexQuery(op="regex", path=self.path, pattern=pattern, flags=flags)

    def type_is(self, obj: str | type | object) -> Query:
        return TypeIsQuery(op="type_is", path=self.path, type=_resolve_type_name(obj))

    def is_a(self, obj: str | type | object) -> Query:
        return IsAQuery(op="is_a", path=self.path, type=_resolve_type_name(obj))

    def related_to(self, obj: str | type | object) -> Query:
        return RelatedToQuery(
            op="related_to", path=self.path, type=_resolve_type_name(obj)
        )


def _resolve_type_name(obj: str | type | object) -> str:
    if isinstance(obj, str):
        return obj

    if not isinstance(obj, type):
        obj = type(obj)

    module = obj.__module__
    qual_name = obj.__qualname__

    if module and module != "builtins":
        return f"{module}.{qual_name}"

    return qual_name


class Q:
    """Entry point for building query expressions in Python."""

    exp = FieldRef("exp")
    config = FieldRef("config")
    meta = FieldRef("meta")
    state = FieldRef("state")


TRUE = TrueQuery(op="true")
FALSE = FalseQuery(op="false")


__all__ = ["FieldRef", "Q", "TRUE", "FALSE"]
