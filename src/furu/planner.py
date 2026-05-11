from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from furu.core import Furu


@dataclass(frozen=True, slots=True)
class Plan:
    artifacts: dict[str, Furu[Any]] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)


def build_plan(roots: Iterable[Furu[Any]]) -> Plan:
    artifacts: dict[str, Furu[Any]] = {}
    edges: dict[str, set[str]] = {}

    def walk(obj: Furu[Any]) -> None:
        object_id = obj.object_id
        if object_id in artifacts:
            return
        artifacts[object_id] = obj
        declared = obj._declared_refs()
        edges[object_id] = {ref.object_id for ref in declared}
        for ref in declared:
            walk(ref)

    for root in roots:
        if not isinstance(root, Furu):
            raise TypeError("build_plan() expected Furu objects")
        walk(root)

    return Plan(artifacts=artifacts, edges=edges)
