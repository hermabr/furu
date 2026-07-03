from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from furu.core import Spec


ExplainDepth: TypeAlias = int | Literal["full"]


def explain(spec: Spec[Any], *, depth: ExplainDepth = 0) -> str:
    return "\n".join(_explain_lines(spec, depth=depth))


def _explain_lines(value: object, *, depth: ExplainDepth) -> list[str]:
    value_fields = _explain_fields(value)
    lines = [_explain_header(value)]
    if not value_fields:
        return lines
    width = max(len(name) for name in value_fields)
    for name in value_fields:
        field_value = getattr(value, name)
        if depth != 0 and _is_explainable(field_value):
            nested = _explain_lines(
                field_value, depth="full" if depth == "full" else depth - 1
            )
            lines.append(f"  {name:<{width}}  {nested[0]}")
            lines.extend(f"  {line}" for line in nested[1:])
        else:
            lines.append(f"  {name:<{width}}  {_format_explain_value(field_value)}")
    return lines


def _explain_header(value: object) -> str:
    from furu.core import Spec

    if isinstance(value, Spec):
        return (
            f"{type(value).__name__}"
            f"  schema={value._artifact_schema_hash[:5]}"
            f"  fields={value._artifact_hash[:5]}"
        )
    return type(value).__name__


def _explain_fields(value: object) -> tuple[str, ...]:
    if isinstance(value, PydanticBaseModel):
        return tuple(type(value).model_fields)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(field.name for field in fields(value))
    return ()


def _is_explainable(value: object) -> bool:
    from furu.core import Spec

    return isinstance(value, (Spec, PydanticBaseModel)) or (
        is_dataclass(value) and not isinstance(value, type)
    )


def _format_explain_value(value: object) -> str:
    from furu.core import Spec

    if isinstance(value, Spec):
        return f"{type(value).__name__} \u00b7 key={value._artifact_hash[:5]}\u2026"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return f"{value:_}"
    return repr(value)
