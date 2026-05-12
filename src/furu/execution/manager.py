from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from furu.core import Furu
from furu.dag import FuruDagNode, make_execution_dag
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.serialize import load_furu_from_artifact


class Job(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )
    lease_id: str
    artifact: ArtifactSpec


type GetJobResponse = Job | Literal["wait", "stop"]


class Manager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.nodes_by_id: dict[str, FuruDagNode[Furu[Any]]] = {}
        self.ready: dict[str, FuruDagNode[Furu[Any]]] = {}
        self.blocked: dict[str, FuruDagNode[Furu[Any]]] = {}
        self.running: dict[str, FuruDagNode[Furu[Any]]] = {}
        self.completed: dict[str, FuruDagNode[Furu[Any]]] = {}
        self.failed: dict[str, FuruDagNode[Furu[Any]]] = {}

    @classmethod
    def from_objs(cls, objs: Sequence[Furu[Any]]) -> Manager:
        manager = cls()
        zero_dep = make_execution_dag(objs, manager.nodes_by_id)
        for node in zero_dep:
            manager.ready[node.obj.object_id] = node
        for object_id, node in manager.nodes_by_id.items():
            if object_id not in manager.ready:
                manager.blocked[object_id] = node
        return manager

    @classmethod
    def submit(
        cls,
        objs: Sequence[Furu[Any]],
        *,
        n_workers: int = 4,
        host: str = "127.0.0.1",
    ) -> Manager:
        from furu.execution.server import run_until_done

        manager = cls.from_objs(objs)
        run_until_done(manager, n_workers=n_workers, host=host)
        return manager

    def get_job(self) -> GetJobResponse:
        with self.lock:
            if self.ready:
                object_id = next(iter(self.ready))
                node = self.ready.pop(object_id)
                lease_id = str(uuid.uuid4())
                self.running[lease_id] = node
                return Job(
                    lease_id=lease_id,
                    artifact=ArtifactSpec.from_furu(node.obj),
                )
            if not self.running:
                return "stop"
            return "wait"

    def finish(self, lease_id: str, *, success: bool) -> None:
        with self.lock:
            node = self.running.pop(lease_id)
            object_id = node.obj.object_id
            if success:
                self.completed[object_id] = node
                for dependent in node.dependents:
                    if node in dependent.dependencies:
                        dependent.dependencies.remove(node)
                    dependent_id = dependent.obj.object_id
                    if not dependent.dependencies and dependent_id in self.blocked:
                        del self.blocked[dependent_id]
                        self.ready[dependent_id] = dependent
            else:
                self.failed[object_id] = node

    def report_blocked(
        self,
        lease_id: str,
        dependencies: Sequence[ArtifactSpec],
    ) -> None:
        with self.lock:
            node = self.running.pop(lease_id)
            dep_objs = [load_furu_from_artifact(spec) for spec in dependencies]
            new_objs = [
                obj for obj in dep_objs if obj.object_id not in self.nodes_by_id
            ]
            new_zero_dep = make_execution_dag(new_objs, self.nodes_by_id)
            new_zero_dep_ids = {n.obj.object_id for n in new_zero_dep}
            for new_node in new_zero_dep:
                self.ready[new_node.obj.object_id] = new_node
            for new_obj in new_objs:
                if new_obj.object_id in new_zero_dep_ids:
                    continue
                if new_obj.object_id in self.ready:
                    continue
                self.blocked[new_obj.object_id] = self.nodes_by_id[new_obj.object_id]

            for dep_obj in dep_objs:
                dep_id = dep_obj.object_id
                if dep_id in self.completed:
                    continue
                dep_node = self.nodes_by_id[dep_id]
                if dep_node not in node.dependencies:
                    node.dependencies.append(dep_node)
                    dep_node.dependents.append(node)

            parent_id = node.obj.object_id
            if not node.dependencies:
                self.ready[parent_id] = node
            else:
                self.blocked[parent_id] = node

    def unresolved_object_ids(self) -> list[str]:
        with self.lock:
            return sorted(self.blocked)

    def log_summary(self) -> None:
        logger = get_logger("manager")
        if self.failed:
            logger.error(
                "manager finished with %d failed node(s): %s",
                len(self.failed),
                sorted(self.failed),
            )
        if self.blocked:
            logger.error(
                "manager finished with %d unresolved node(s): %s",
                len(self.blocked),
                sorted(self.blocked),
            )
        if not self.failed and not self.blocked:
            logger.info(
                "manager finished successfully (%d node(s) completed)",
                len(self.completed),
            )
