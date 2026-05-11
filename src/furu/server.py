from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from furu.core import Furu
from furu.graph import (
    ArtifactNode,
    DiscoveredGraph,
    NodeKey,
    discover_missing_closure,
)
from furu.metadata import ArtifactSpec
from furu.migration import result_dir_for_loading
from furu.submission import SubmissionStatus


class CreateSubmissionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    graph: DiscoveredGraph
    roots: tuple[NodeKey, ...]
    input_order: tuple[NodeKey, ...]
    single_input: bool


class CreateSubmissionResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    submission_id: str


class LeaseResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    lease_id: str
    submission_id: str
    node: ArtifactNode


class LeaseDone(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    kind: Literal["done"]
    node_key: NodeKey


class LeaseDependencyNotReady(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    kind: Literal["dependency_not_ready"]
    blocked: NodeKey
    call_kind: Literal["load_or_create", "try_load"]
    dependencies: tuple[ArtifactNode, ...]


class LeaseFailed(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    kind: Literal["failed"]
    node_key: NodeKey
    error_type: str
    error_message: str
    traceback: str


LeaseResult = Annotated[
    LeaseDone | LeaseDependencyNotReady | LeaseFailed,
    Field(discriminator="kind"),
]


class LeaseResultEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    result: LeaseResult


@dataclass(frozen=True, kw_only=True)
class FailureRecord:
    error_type: str
    error_message: str
    traceback: str


class NodeRunState(Enum):
    QUEUED = "queued"
    RUNNING = "running"


@dataclass
class NodeRecord:
    key: NodeKey
    artifact: ArtifactSpec
    dependencies: set[NodeKey] = field(default_factory=set)
    dependents: set[NodeKey] = field(default_factory=set)
    state: NodeRunState | None = None
    active_lease_id: str | None = None
    failure: FailureRecord | None = None


@dataclass
class SubmissionRecord:
    id: str
    roots: tuple[NodeKey, ...]
    input_order: tuple[NodeKey, ...]
    single_input: bool


@dataclass
class LeaseRecord:
    lease_id: str
    submission_id: str
    node: ArtifactNode


class Launcher:
    def capacity(self) -> int:
        raise NotImplementedError

    def launch(self, leases: list[LeaseRecord]) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


def _node_is_done(node: NodeRecord) -> bool:
    obj = Furu.from_artifact(node.artifact)
    return result_dir_for_loading(obj) is not None


class Scheduler:
    def __init__(self, *, launcher: Launcher) -> None:
        self.lock = threading.RLock()
        self.submissions: dict[str, SubmissionRecord] = {}
        self.nodes: dict[NodeKey, NodeRecord] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self.node_to_submission: dict[NodeKey, str] = {}
        self.job_groups: dict[str, tuple[str, ...]] = {}
        self.launcher = launcher

    def create_submission(self, req: CreateSubmissionRequest) -> str:
        submission_id = uuid.uuid4().hex

        with self.lock:
            self.submissions[submission_id] = SubmissionRecord(
                id=submission_id,
                roots=req.roots,
                input_order=req.input_order,
                single_input=req.single_input,
            )

            self._merge_graph(req.graph)

            for node_key in self._reachable_from_roots(req.roots):
                self.node_to_submission.setdefault(node_key, submission_id)

            self._recompute_queue_state()

        return submission_id

    def _reachable_from_roots(self, roots: Iterable[NodeKey]) -> set[NodeKey]:
        reachable: set[NodeKey] = set()
        stack = list(roots)
        while stack:
            key = stack.pop()
            if key in reachable:
                continue
            reachable.add(key)
            node = self.nodes.get(key)
            if node is None:
                continue
            stack.extend(node.dependencies)
        return reachable

    def _merge_graph(self, graph: DiscoveredGraph) -> None:
        for artifact_node in graph.nodes:
            if artifact_node.key not in self.nodes:
                self.nodes[artifact_node.key] = NodeRecord(
                    key=artifact_node.key,
                    artifact=artifact_node.artifact,
                )

        for dependency, dependent in graph.edges:
            self._add_edge(dependency=dependency, dependent=dependent)

    def _add_edge(self, *, dependency: NodeKey, dependent: NodeKey) -> None:
        if dependency == dependent:
            self.nodes[dependent].failure = FailureRecord(
                error_type="DependencyCycle",
                error_message="Artifact depends on itself",
                traceback="",
            )
            return

        self.nodes[dependent].dependencies.add(dependency)
        self.nodes[dependency].dependents.add(dependent)

    def _is_done(self, node: NodeRecord) -> bool:
        return _node_is_done(node)

    def _has_failed_dependency(self, node: NodeRecord) -> bool:
        return any(self.nodes[dep].failure is not None for dep in node.dependencies)

    def _deps_done(self, node: NodeRecord) -> bool:
        return all(self._is_done(self.nodes[dep]) for dep in node.dependencies)

    def _recompute_queue_state(self) -> None:
        for node in self.nodes.values():
            # Active leases are always RUNNING; never clear them here.
            # The lease-result handlers are the only callers that may
            # clear node.active_lease_id, so the worker's result post
            # is never raced against by recomputation.
            if node.active_lease_id is not None:
                node.state = NodeRunState.RUNNING
                continue

            if self._is_done(node):
                node.state = None
                continue

            if node.failure is not None:
                node.state = None
                continue

            if self._has_failed_dependency(node):
                node.state = None
                continue

            if self._deps_done(node):
                node.state = NodeRunState.QUEUED
            else:
                node.state = None

    def running_count(self) -> int:
        return sum(
            1 for node in self.nodes.values() if node.state == NodeRunState.RUNNING
        )

    def _submission_for_node(self, node_key: NodeKey) -> str:
        submission_id = self.node_to_submission.get(node_key)
        if submission_id is None:
            raise RuntimeError(f"node {node_key} not associated with a submission")
        return submission_id

    def _reserve_lease(self, node: NodeRecord) -> LeaseRecord:
        assert node.state == NodeRunState.QUEUED
        assert node.active_lease_id is None

        lease_id = uuid.uuid4().hex

        lease = LeaseRecord(
            lease_id=lease_id,
            submission_id=self._submission_for_node(node.key),
            node=ArtifactNode(
                key=node.key,
                artifact=node.artifact,
            ),
        )

        self.leases[lease_id] = lease

        node.active_lease_id = lease_id
        node.state = NodeRunState.RUNNING

        return lease

    def schedule_queued_nodes(self) -> None:
        leases: list[LeaseRecord] = []
        with self.lock:
            self._recompute_queue_state()

            capacity = self.launcher.capacity() - self.running_count()

            if capacity <= 0:
                return

            queued_nodes = [
                node
                for node in self.nodes.values()
                if node.state == NodeRunState.QUEUED
            ]

            for node in queued_nodes[:capacity]:
                leases.append(self._reserve_lease(node))

        if leases:
            self.launcher.launch(leases)

    def get_lease(self, lease_id: str) -> LeaseRecord:
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                raise KeyError(f"unknown lease: {lease_id}")
            return lease

    def get_lease_for_array_index(
        self,
        *,
        job_group_id: str,
        array_index: int,
    ) -> LeaseRecord:
        with self.lock:
            lease_ids = self.job_groups.get(job_group_id)
            if lease_ids is None:
                raise KeyError(f"unknown job group: {job_group_id}")
            if not 0 <= array_index < len(lease_ids):
                raise KeyError(
                    f"array index {array_index} out of range for job group "
                    f"{job_group_id} (size {len(lease_ids)})"
                )
            lease = self.leases.get(lease_ids[array_index])
            if lease is None:
                raise KeyError(f"lease for array index {array_index} no longer exists")
            return lease

    def create_job_group(self, leases: list[LeaseRecord]) -> str:
        job_group_id = uuid.uuid4().hex
        with self.lock:
            self.job_groups[job_group_id] = tuple(lease.lease_id for lease in leases)
        return job_group_id

    def _validate_lease(self, lease_id: str) -> LeaseRecord:
        lease = self.leases.get(lease_id)
        if lease is None:
            raise KeyError(f"unknown lease: {lease_id}")
        return lease

    def _clear_lease(self, node: NodeRecord, lease_id: str) -> None:
        if node.active_lease_id != lease_id:
            raise RuntimeError("lease does not match active node lease")
        node.active_lease_id = None
        node.state = None
        self.leases.pop(lease_id, None)

    def handle_lease_result(
        self,
        lease_id: str,
        result: LeaseDone | LeaseDependencyNotReady | LeaseFailed,
    ) -> None:
        if isinstance(result, LeaseDone):
            self._handle_lease_done(lease_id, result)
        elif isinstance(result, LeaseDependencyNotReady):
            self._handle_lease_dependency(lease_id, result)
        elif isinstance(result, LeaseFailed):
            self._handle_lease_failed(lease_id, result)

        self.schedule_queued_nodes()

    def _handle_lease_done(self, lease_id: str, result: LeaseDone) -> None:
        with self.lock:
            self._validate_lease(lease_id)
            node = self.nodes[result.node_key]

            self._clear_lease(node, lease_id)

            if not self._is_done(node):
                node.failure = FailureRecord(
                    error_type="MissingResult",
                    error_message=("Worker reported success but result does not exist"),
                    traceback="",
                )

            self._recompute_queue_state()

    def _handle_lease_dependency(
        self,
        lease_id: str,
        result: LeaseDependencyNotReady,
    ) -> None:
        with self.lock:
            self._validate_lease(lease_id)
            blocked = self.nodes[result.blocked]
            submission_id = self.node_to_submission.get(blocked.key)
            self._clear_lease(blocked, lease_id)

        dependency_objects = [
            Furu.from_artifact(dep.artifact) for dep in result.dependencies
        ]

        graph = discover_missing_closure(dependency_objects)

        with self.lock:
            self._merge_graph(graph)

            for dep in result.dependencies:
                self._add_edge(
                    dependency=dep.key,
                    dependent=result.blocked,
                )

            if submission_id is not None:
                for key in self._reachable_from_roots([result.blocked]):
                    self.node_to_submission.setdefault(key, submission_id)

            self._recompute_queue_state()

    def _handle_lease_failed(self, lease_id: str, result: LeaseFailed) -> None:
        with self.lock:
            self._validate_lease(lease_id)
            node = self.nodes[result.node_key]

            self._clear_lease(node, lease_id)

            node.failure = FailureRecord(
                error_type=result.error_type,
                error_message=result.error_message,
                traceback=result.traceback,
            )

            self._recompute_queue_state()

    def submission_status(self, submission_id: str) -> SubmissionStatus:
        with self.lock:
            submission = self.submissions.get(submission_id)
            if submission is None:
                raise KeyError(f"unknown submission: {submission_id}")

            reachable = self._reachable_from_roots(submission.roots)

            total = len(reachable)
            done = 0
            queued = 0
            running = 0
            failure_message: str | None = None
            has_failure = False

            for key in reachable:
                node = self.nodes.get(key)
                if node is None:
                    continue
                if self._is_done(node):
                    done += 1
                elif node.state == NodeRunState.QUEUED:
                    queued += 1
                elif node.state == NodeRunState.RUNNING:
                    running += 1

                if node.failure is not None and failure_message is None:
                    has_failure = True
                    failure_message = (
                        f"{node.failure.error_type}: {node.failure.error_message}"
                    )

            if has_failure:
                status: Literal["running", "done", "failed"] = "failed"
            elif done == total:
                status = "done"
            else:
                status = "running"

            return SubmissionStatus(
                submission_id=submission_id,
                status=status,
                failure_message=failure_message,
                total_nodes=total,
                done_nodes=done,
                queued_nodes=queued,
                running_nodes=running,
            )


def build_app(*, scheduler: Scheduler, token: str) -> FastAPI:
    app = FastAPI()

    def require_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = f"Bearer {token}"
        if authorization != expected:
            raise HTTPException(
                status_code=401, detail="invalid or missing bearer token"
            )

    AuthDep = Depends(require_token)

    @app.post("/api/v1/submissions", dependencies=[AuthDep])
    async def create_submission(request: Request) -> CreateSubmissionResponse:
        body = await request.body()
        req = CreateSubmissionRequest.model_validate_json(body)
        submission_id = scheduler.create_submission(req)
        scheduler.schedule_queued_nodes()
        return CreateSubmissionResponse(submission_id=submission_id)

    @app.get("/api/v1/submissions/{submission_id}", dependencies=[AuthDep])
    def get_submission_status(submission_id: str) -> SubmissionStatus:
        try:
            return scheduler.submission_status(submission_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/leases/{lease_id}", dependencies=[AuthDep])
    def get_lease(lease_id: str) -> LeaseResponse:
        try:
            lease = scheduler.get_lease(lease_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return LeaseResponse(
            lease_id=lease.lease_id,
            submission_id=lease.submission_id,
            node=lease.node,
        )

    @app.post("/api/v1/leases/{lease_id}/result", dependencies=[AuthDep])
    async def post_lease_result(lease_id: str, request: Request) -> dict[str, str]:
        body = await request.body()
        envelope = LeaseResultEnvelope.model_validate_json(body)
        try:
            scheduler.handle_lease_result(lease_id, envelope.result)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.get(
        "/api/v1/job-groups/{job_group_id}/leases/{array_index}",
        dependencies=[AuthDep],
    )
    def get_lease_for_array_index(job_group_id: str, array_index: int) -> LeaseResponse:
        try:
            lease = scheduler.get_lease_for_array_index(
                job_group_id=job_group_id,
                array_index=array_index,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return LeaseResponse(
            lease_id=lease.lease_id,
            submission_id=lease.submission_id,
            node=lease.node,
        )

    return app


class SchedulerClient:
    def __init__(
        self,
        *,
        http_client: httpx.Client,
        owns_client: bool = True,
    ) -> None:
        self._client = http_client
        self._owns_client = owns_client

    @classmethod
    def for_remote(cls, *, base_url: str, token: str) -> "SchedulerClient":
        return cls(
            http_client=httpx.Client(
                base_url=base_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0,
            ),
            owns_client=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SchedulerClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def create_submission(
        self, req: CreateSubmissionRequest
    ) -> CreateSubmissionResponse:
        response = self._client.post(
            "/api/v1/submissions",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return CreateSubmissionResponse.model_validate_json(response.text)

    def get_submission_status(self, submission_id: str) -> SubmissionStatus:
        response = self._client.get(f"/api/v1/submissions/{submission_id}")
        response.raise_for_status()
        return SubmissionStatus.model_validate_json(response.text)

    def get_lease(self, lease_id: str) -> LeaseResponse:
        response = self._client.get(f"/api/v1/leases/{lease_id}")
        response.raise_for_status()
        return LeaseResponse.model_validate_json(response.text)

    def get_lease_for_array_index(
        self, *, job_group_id: str, array_index: int
    ) -> LeaseResponse:
        response = self._client.get(
            f"/api/v1/job-groups/{job_group_id}/leases/{array_index}"
        )
        response.raise_for_status()
        return LeaseResponse.model_validate_json(response.text)

    def post_lease_result(
        self,
        *,
        lease_id: str,
        result: LeaseDone | LeaseDependencyNotReady | LeaseFailed,
    ) -> None:
        envelope = LeaseResultEnvelope(result=result)
        response = self._client.post(
            f"/api/v1/leases/{lease_id}/result",
            content=envelope.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()


def build_test_client(*, app: FastAPI, token: str) -> httpx.Client:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {token}"
    return client
