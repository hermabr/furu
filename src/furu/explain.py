from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from furu._pytree import tree_node

if TYPE_CHECKING:
    from furu.core import Spec


type ExplainDepth = int | Literal["full"]


def explain(spec: Spec[Any], *, depth: ExplainDepth = 0) -> str:
    header = (
        f"{type(spec).__name__}"
        f"  schema={spec._artifact_schema_hash[:5]}"
        f"  key={spec._artifact_hash[:5]}…"
    )
    return "\n".join([header, *_rows(_children(spec) or [], depth)])


def _rows(children: list[tuple[str, object]], depth: ExplainDepth) -> list[str]:
    width = max((len(name) for name, _ in children), default=0)
    lines: list[str] = []
    for name, value in children:
        from furu.core import Spec

        if isinstance(value, Spec) and depth == 0:
            rendered = [f"{type(value).__name__} · key={value._artifact_hash[:5]}…"]
        else:
            child_values = _children(value)
            if child_values is None:
                match value:
                    case str():
                        rendered = [json.dumps(value)]
                    case bool():
                        rendered = [repr(value)]
                    case int():
                        rendered = [f"{value:_}"]
                    case float() | None | Path() | type():
                        rendered = [repr(value)]
                    case _:
                        rendered = [f"custom · {value!r}"]
            elif depth == 0:
                rendered = [repr(value)]
            else:
                rendered = [
                    f"{type(value).__name__} · key={value._artifact_hash[:5]}…"
                    if isinstance(value, Spec)
                    else type(value).__name__,
                    *_rows(
                        child_values,
                        "full" if depth == "full" else depth - 1,
                    ),
                ]
        lines.append(f"  {name:<{width}}  {rendered[0]}")
        lines.extend(f"  {line}" for line in rendered[1:])
    return lines


def _children(value: object) -> list[tuple[str, object]] | None:
    if (node := tree_node(value)) is None:
        return None
    return [
        (str(entry), child)
        for entry, child in zip(node.entries, node.children, strict=True)
    ]
