from __future__ import annotations

import json
from dataclasses import fields
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from furu.core import Spec


ExplainDepth: TypeAlias = int | Literal["full"]


def explain(spec: Spec[Any], *, depth: ExplainDepth = 0) -> str:
    return "\n".join(_explain_lines(spec, depth=depth))


def _explain_lines(spec: Spec[Any], *, depth: ExplainDepth) -> list[str]:
    from furu.core import Spec

    spec_fields = fields(spec)
    lines = [
        f"{type(spec).__name__}"
        f"  schema={spec._artifact_schema_hash[:5]}"
        f"  fields={spec._artifact_hash[:5]}"
    ]
    if not spec_fields:
        return lines
    width = max(len(field.name) for field in spec_fields)
    for field in spec_fields:
        value = getattr(spec, field.name)
        if depth != 0 and isinstance(value, Spec):
            nested = _explain_lines(
                value, depth="full" if depth == "full" else depth - 1
            )
            lines.append(f"  {field.name:<{width}}  {nested[0]}")
            lines.extend(f"  {line}" for line in nested[1:])
        else:
            lines.append(f"  {field.name:<{width}}  {_format_explain_value(value)}")
    return lines


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
