from __future__ import annotations

import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import uvicorn

from furu.core import Furu
from furu.dag import FuruDagNode, make_execution_dag
from furu.execution import _load_or_create_local
from furu.metadata import ArtifactSpec
from furu.serialize import _load_type
from furu.worker_execution import _DependencyNotReady, worker_execution_context

logger = logging.getLogger(__name__)


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lease_id: str
    artifact: ArtifactSpec


class FinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    success: bool


class BlockedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dependencies: list[ArtifactSpec]


type JobResponse = Job | Literal["wait", "stop"]


@dataclass
class Manager:
    nodes_by_id: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    ready: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    blocked: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    running: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    completed: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    failed: dict[str, FuruDagNode[Furu[Any]]] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    done: threading.Event = field(default_factory=threading.Event)

    @classmethod
    def submit(cls, objs: Sequence[Furu[Any]], *, n_workers: int = 1) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        manager = cls.from_objects(objs)
        manager.run_server_workers(n_workers=n_workers)
        manager.raise_if_failed()

    @classmethod
    def from_objects(cls, objs: Sequence[Furu[Any]]) -> Manager:
        manager = cls()
        make_execution_dag(
            objs,
            manager.nodes_by_id,
            ready=manager.ready,
            blocked=manager.blocked,
        )
        manager._check_finished_locked()
        return manager

    def get_job(self) -> JobResponse:
        with self.lock:
            if self.done.is_set():
                return "stop"
            if not self.ready:
                return "wait"
            lease_id, node = self.ready.popitem()
            self.running[lease_id] = node
            return Job(lease_id=lease_id, artifact=ArtifactSpec.from_furu(node.obj))

    def finish(self, lease_id: str, *, success: bool) -> None:
        with self.lock:
            node = self.running.pop(lease_id, None)
            if node is None:
                raise KeyError(f"unknown running lease {lease_id!r}")

            if success:
                self.completed[lease_id] = node
                self.nodes_by_id.pop(lease_id, None)
                self._release_dependents_locked(node)
            else:
                self.failed[lease_id] = node

            self._check_finished_locked()

    def block(self, lease_id: str, dependencies: Sequence[ArtifactSpec]) -> None:
        with self.lock:
            node = self.running.pop(lease_id, None)
            if node is None:
                raise KeyError(f"unknown running lease {lease_id!r}")
            self.blocked[lease_id] = node

            new_objs = [
                artifact_to_furu(dep)
                for dep in dependencies
                if dep.object_id not in self.nodes_by_id
            ]
            make_execution_dag(
                new_objs,
                self.nodes_by_id,
                ready=self.ready,
                blocked=self.blocked,
            )
            for dep in dependencies:
                dep_node = self.nodes_by_id[dep.object_id]
                if dep_node not in node.dependencies:
                    node.dependencies.append(dep_node)
                    dep_node.dependents.append(node)

            self._check_finished_locked()

    def run_workers(self, *, n_workers: int) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        threads = [
            threading.Thread(target=run_worker_against_manager, args=(self,))
            for _ in range(n_workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def run_server_workers(self, *, n_workers: int) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        if self.done.is_set():
            return

        host = "127.0.0.1"
        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(self),
                host=host,
                port=port,
                lifespan="off",
                log_level="warning",
            )
        )
        server_thread = threading.Thread(target=server.run)
        server_thread.start()

        base_url = f"http://{host}:{port}"
        worker_threads = [
            threading.Thread(target=worker_loop, args=(base_url,))
            for _ in range(n_workers)
        ]
        for thread in worker_threads:
            thread.start()

        self.done.wait()
        for thread in worker_threads:
            thread.join()
        server.should_exit = True
        server_thread.join()

    def raise_if_failed(self) -> None:
        with self.lock:
            if self.failed:
                failed = ", ".join(sorted(self.failed))
                raise RuntimeError(f"submit() failed jobs: {failed}")
            if self.blocked:
                blocked = ", ".join(sorted(self.blocked))
                raise RuntimeError(
                    "submit() could not make progress; unresolved dependencies: "
                    f"{blocked}"
                )

    def _release_dependents_locked(self, node: FuruDagNode[Furu[Any]]) -> None:
        for dependent in node.dependents:
            if node in dependent.dependencies:
                dependent.dependencies.remove(node)
            if not dependent.dependencies:
                object_id = dependent.obj.object_id
                if object_id in self.blocked:
                    self.blocked.pop(object_id)
                    self.ready[object_id] = dependent

    def _check_finished_locked(self) -> None:
        if self.running or self.ready:
            return
        if self.blocked or self.failed:
            logger.error(
                "worker manager stopped with blocked=%d failed=%d",
                len(self.blocked),
                len(self.failed),
            )
        else:
            logger.info("worker manager finished")
        self.done.set()


def artifact_to_furu(artifact: ArtifactSpec) -> Furu[Any]:
    cls = _load_type(artifact.fully_qualified_name)
    if not issubclass(cls, Furu):
        raise TypeError(f"{artifact.fully_qualified_name!r} is not a Furu subclass")
    return cls.from_artifact(artifact)


def run_worker_against_manager(manager: Manager) -> None:
    while True:
        response = manager.get_job()
        if response == "stop":
            return
        if response == "wait":
            time.sleep(0.01)
            continue
        _run_job(response, manager)


def _run_job(job: Job, manager: Manager) -> None:
    obj = artifact_to_furu(job.artifact)
    try:
        with worker_execution_context(lease_id=job.lease_id):
            _load_or_create_local(obj)
    except _DependencyNotReady as exc:
        manager.block(
            job.lease_id,
            [ArtifactSpec.from_furu(dep) for dep in exc.dependencies],
        )
    except BaseException:
        manager.finish(job.lease_id, success=False)
        raise
    else:
        manager.finish(job.lease_id, success=True)


def create_app(manager: Manager) -> FastAPI:
    app = FastAPI()

    @app.get("/get_job", response_model=Job | Literal["wait", "stop"])
    def get_job() -> JobResponse:
        return manager.get_job()

    @app.post("/finish/{lease_id}")
    def finish(lease_id: str, request: FinishRequest) -> dict[str, str]:
        try:
            manager.finish(lease_id, success=request.success)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/blocked/{lease_id}")
    def blocked(lease_id: str, request: BlockedRequest) -> dict[str, str]:
        try:
            manager.block(lease_id, request.dependencies)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "ok"}

    return app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker_loop(base_url: str) -> None:
    deadline: float | None = None
    while True:
        try:
            response = _request_json("GET", f"{base_url}/get_job")
            deadline = None
        except urllib.error.URLError:
            if deadline is None:
                deadline = time.monotonic() + 30
            if time.monotonic() >= deadline:
                return
            time.sleep(5)
            continue

        if response == "stop":
            return
        if response == "wait":
            time.sleep(0.1)
            continue

        job = Job.model_validate(response)
        obj = artifact_to_furu(job.artifact)
        try:
            with worker_execution_context(lease_id=job.lease_id):
                _load_or_create_local(obj)
        except _DependencyNotReady as exc:
            _request_json(
                "POST",
                f"{base_url}/blocked/{job.lease_id}",
                BlockedRequest(
                    dependencies=[
                        ArtifactSpec.from_furu(dep) for dep in exc.dependencies
                    ]
                ).model_dump(mode="json"),
            )
        except BaseException:
            _request_json(
                "POST",
                f"{base_url}/finish/{job.lease_id}",
                FinishRequest(success=False).model_dump(mode="json"),
            )
            raise
        else:
            _request_json(
                "POST",
                f"{base_url}/finish/{job.lease_id}",
                FinishRequest(success=True).model_dump(mode="json"),
            )


def _request_json(method: str, url: str, body: object | None = None) -> object:
    import json

    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = response.read()
    return json.loads(payload)
