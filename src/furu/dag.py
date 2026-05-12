from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from furu.dependencies import collect_declared_refs

if TYPE_CHECKING:
    from furu.core import Furu


@dataclass(frozen=True, eq=False)
class FuruDependencyNode[TFuru: Furu]:
    obj: TFuru
    dependencies: tuple[FuruDependencyNode[Furu], ...] = field(
        default=(),
        repr=False,
    )
    dependents: tuple[FuruDependencyNode[Furu], ...] = field(
        default=(),
        repr=False,
    )


def _normalize_execution_dag_input[TFuru: Furu](objs: list[TFuru]) -> list[TFuru]:
    from furu.core import Furu

    if not isinstance(objs, list):
        raise TypeError("make_execution_dag() expected a list of Furu objects")
    if any(not isinstance(obj, Furu) for obj in objs):
        raise TypeError("make_execution_dag() expected Furu objects")
    return objs


def _is_success_status[TFuru: Furu](obj: TFuru) -> bool:
    return obj.status() == "completed"


def make_execution_dag[TFuru: Furu](
    objs: list[TFuru],
) -> tuple[list[Furu], dict[str, FuruDependencyNode[Furu]]]:
    objs = _normalize_execution_dag_input(objs)

    nodes_by_id: dict[str, FuruDependencyNode[Furu]] = {}
    dependency_ids_by_id: dict[str, tuple[str, ...]] = {}
    dependent_ids_by_id: dict[str, list[str]] = {}
    visiting: set[str] = set()

    def get_node(obj: Furu) -> FuruDependencyNode[Furu]:
        obj_id = obj.object_id
        if obj_id in nodes_by_id:
            return nodes_by_id[obj_id]
        node = FuruDependencyNode(obj=obj)
        nodes_by_id[obj_id] = node
        return node

    def visit(obj: Furu) -> None:
        obj_id = obj.object_id
        get_node(obj)
        if obj_id in visiting:
            raise ValueError(f"declared Furu dependencies contain a cycle at {obj_id}")
        if obj_id in dependency_ids_by_id:
            return

        visiting.add(obj_id)
        refs = () if _is_success_status(obj) else collect_declared_refs(obj)
        for ref in refs:
            visit(ref)
            dependent_ids_by_id.setdefault(ref.object_id, []).append(obj_id)
        dependency_ids_by_id[obj_id] = tuple(ref.object_id for ref in refs)
        visiting.remove(obj_id)

    for obj in objs:
        visit(obj)

    for obj_id, node in nodes_by_id.items():
        object.__setattr__(
            node,
            "dependencies",
            tuple(
                nodes_by_id[dependency_id]
                for dependency_id in dependency_ids_by_id[obj_id]
            ),
        )
        object.__setattr__(
            node,
            "dependents",
            tuple(
                nodes_by_id[dependent_id]
                for dependent_id in dependent_ids_by_id.get(obj_id, [])
            ),
        )

    return (
        [node.obj for node in nodes_by_id.values() if not node.dependencies],
        nodes_by_id,
    )
