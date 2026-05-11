from __future__ import annotations

import secrets
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, Protocol

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_validator

from furu.core import Furu
from furu.graph import ArtifactNode, DiscoveredGraph, NodeKey, discover_missing_closure
from furu.migration import result_dir_for_loading


class NodeRunState(Enum):
    QUEUED = "queued"
    RUNNING = "running"


class FailureRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    error_type: str
    error_message: str
    traceback: str


@dataclass
class NodeRecord:
    key: NodeKey
    artifact: Any
    dependencies: set[NodeKey] = field(default_factory=set)
    dependents: set[NodeKey] = field(default_factory=set)
    state: NodeRunState | None = None
    active_lease_id: str | None = None
    failure: FailureRecord | None = None


@dataclass(frozen=True)
class SubmissionRecord:
    id: str
    roots: tuple[NodeKey, ...]
    input_order: tuple[NodeKey, ...]
    single_input: bool


@dataclass(frozen=True)
class LeaseRecord:
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

    @field_validator("dependencies", mode="before")
    @classmethod
    def _coerce_dependencies(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


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


class CreateSubmissionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    graph: DiscoveredGraph
    roots: tuple[NodeKey, ...]
    input_order: tuple[NodeKey, ...]
    single_input: bool

    @field_validator("roots", "input_order", mode="before")
    @classmethod
    def _coerce_json_arrays(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class CreateSubmissionResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    submission_id: str


class SubmissionStatusResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    submission_id: str
    status: Literal["running", "done", "failed"]
    failure_message: str | None = None
    total_nodes: int
    done_nodes: int
    queued_nodes: int
    running_nodes: int


class LeaseResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    lease_id: str
    submission_id: str
    node: ArtifactNode


class Launcher(Protocol):
    def capacity(self) -> int: ...

    def launch(self, leases: list[LeaseRecord]) -> None: ...


def node_object(node: NodeRecord) -> Furu[Any]:
    return Furu.from_artifact(node.artifact)


class Scheduler:
    def __init__(self, *, launcher: Launcher) -> None:
        self.lock = threading.RLock()
        self.submissions: dict[str, SubmissionRecord] = {}
        self.nodes: dict[NodeKey, NodeRecord] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self.job_groups: dict[str, tuple[str, ...]] = {}
        self.launcher = launcher

    def is_done(self, node: NodeRecord) -> bool:
        return result_dir_for_loading(node_object(node)) is not None

    def has_failed_dependency(self, node: NodeRecord) -> bool:
        return any(self.nodes[dep].failure is not None for dep in node.dependencies)

    def deps_done(self, node: NodeRecord) -> bool:
        return all(self.is_done(self.nodes[dep]) for dep in node.dependencies)

    def merge_graph(self, graph: DiscoveredGraph) -> None:
        for artifact_node in graph.nodes:
            self.nodes.setdefault(
                artifact_node.key,
                NodeRecord(key=artifact_node.key, artifact=artifact_node.artifact),
            )
        for dependency, dependent in graph.edges:
            self.add_edge(dependency=dependency, dependent=dependent)
        self.recompute_queue_state()

    def add_edge(self, *, dependency: NodeKey, dependent: NodeKey) -> None:
        if dependency == dependent:
            self.nodes[dependent].failure = FailureRecord(
                error_type="DependencyCycle",
                error_message="Artifact depends on itself",
                traceback="",
            )
            return
        self.nodes[dependent].dependencies.add(dependency)
        self.nodes[dependency].dependents.add(dependent)

    def recompute_queue_state(self) -> None:
        for node in self.nodes.values():
            if self.is_done(node) or node.failure is not None:
                node.state = None
                node.active_lease_id = None
                continue
            if self.has_failed_dependency(node):
                node.state = None
                continue
            if node.active_lease_id is not None:
                node.state = NodeRunState.RUNNING
                continue
            node.state = NodeRunState.QUEUED if self.deps_done(node) else None

    def create_submission(self, req: CreateSubmissionRequest) -> str:
        with self.lock:
            submission_id = uuid.uuid4().hex
            self.merge_graph(req.graph)
            self.submissions[submission_id] = SubmissionRecord(
                id=submission_id,
                roots=req.roots,
                input_order=req.input_order,
                single_input=req.single_input,
            )
            self.recompute_queue_state()
            return submission_id

    def submission_for_node(self, key: NodeKey) -> str:
        for submission in self.submissions.values():
            if key in submission.roots or key in self.nodes:
                return submission.id
        raise RuntimeError("node is not associated with a submission")

    def running_count(self) -> int:
        return sum(node.active_lease_id is not None for node in self.nodes.values())

    def reserve_lease(self, node: NodeRecord) -> LeaseRecord:
        assert node.state == NodeRunState.QUEUED
        assert node.active_lease_id is None
        lease_id = uuid.uuid4().hex
        lease = LeaseRecord(
            lease_id=lease_id,
            submission_id=self.submission_for_node(node.key),
            node=ArtifactNode(key=node.key, artifact=node.artifact),
        )
        self.leases[lease_id] = lease
        node.active_lease_id = lease_id
        node.state = NodeRunState.RUNNING
        return lease

    def schedule_queued_nodes(self) -> None:
        with self.lock:
            self.recompute_queue_state()
            capacity = self.launcher.capacity() - self.running_count()
            if capacity <= 0:
                return
            queued = [
                node
                for node in self.nodes.values()
                if node.state == NodeRunState.QUEUED
            ]
            leases = [self.reserve_lease(node) for node in queued[:capacity]]
        if leases:
            self.launcher.launch(leases)

    def create_job_group(self, leases: list[LeaseRecord]) -> str:
        job_group_id = uuid.uuid4().hex
        self.job_groups[job_group_id] = tuple(lease.lease_id for lease in leases)
        return job_group_id

    def validate_lease(self, lease_id: str) -> LeaseRecord:
        try:
            return self.leases[lease_id]
        except KeyError as exc:
            raise RuntimeError("unknown lease") from exc

    def clear_lease(self, node: NodeRecord, lease_id: str) -> None:
        if node.active_lease_id != lease_id:
            raise RuntimeError("lease does not match active node lease")
        node.active_lease_id = None
        node.state = None
        self.leases.pop(lease_id, None)

    def handle_lease_result(self, lease_id: str, result: LeaseResult) -> None:
        if isinstance(result, LeaseDone):
            self.handle_lease_result_done(lease_id, result)
        elif isinstance(result, LeaseDependencyNotReady):
            self.handle_lease_result_dependency(lease_id, result)
        else:
            self.handle_lease_result_failed(lease_id, result)

    def handle_lease_result_done(self, lease_id: str, result: LeaseDone) -> None:
        with self.lock:
            self.validate_lease(lease_id)
            node = self.nodes[result.node_key]
            if node.active_lease_id == lease_id:
                self.clear_lease(node, lease_id)
            elif self.is_done(node):
                self.leases.pop(lease_id, None)
            else:
                self.clear_lease(node, lease_id)
            if not self.is_done(node):
                node.failure = FailureRecord(
                    error_type="MissingResult",
                    error_message="Worker reported success but result does not exist",
                    traceback="",
                )
            self.recompute_queue_state()
        self.schedule_queued_nodes()

    def handle_lease_result_dependency(
        self, lease_id: str, result: LeaseDependencyNotReady
    ) -> None:
        with self.lock:
            self.validate_lease(lease_id)
            blocked = self.nodes[result.blocked]
            self.clear_lease(blocked, lease_id)
            blocked.state = None

        graph = discover_missing_closure(
            [Furu.from_artifact(dep.artifact) for dep in result.dependencies]
        )
        with self.lock:
            self.merge_graph(graph)
            for dep in result.dependencies:
                self.add_edge(dependency=dep.key, dependent=result.blocked)
            self.recompute_queue_state()
        self.schedule_queued_nodes()

    def handle_lease_result_failed(self, lease_id: str, result: LeaseFailed) -> None:
        with self.lock:
            self.validate_lease(lease_id)
            node = self.nodes[result.node_key]
            self.clear_lease(node, lease_id)
            node.failure = FailureRecord(
                error_type=result.error_type,
                error_message=result.error_message,
                traceback=result.traceback,
            )
            node.state = None
            self.recompute_queue_state()
        self.schedule_queued_nodes()

    def status(self, submission_id: str) -> SubmissionStatusResponse:
        with self.lock:
            submission = self.submissions[submission_id]
            reachable = self._reachable_from_roots(submission.roots)
            nodes = [self.nodes[key] for key in reachable]
            failures = [
                node.failure
                for node in nodes
                if node.failure is not None or self.has_failed_dependency(node)
            ]
            if failures:
                failure = next((f for f in failures if f is not None), None)
                message = (
                    f"{failure.error_type}: {failure.error_message}"
                    if failure is not None
                    else "dependency failed"
                )
                status: Literal["running", "done", "failed"] = "failed"
            elif all(self.is_done(self.nodes[root]) for root in submission.roots):
                message = None
                status = "done"
            else:
                message = None
                status = "running"
            return SubmissionStatusResponse(
                submission_id=submission_id,
                status=status,
                failure_message=message,
                total_nodes=len(nodes),
                done_nodes=sum(self.is_done(node) for node in nodes),
                queued_nodes=sum(node.state == NodeRunState.QUEUED for node in nodes),
                running_nodes=sum(node.state == NodeRunState.RUNNING for node in nodes),
            )

    def _reachable_from_roots(self, roots: tuple[NodeKey, ...]) -> set[NodeKey]:
        seen: set[NodeKey] = set()
        stack = list(roots)
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            stack.extend(self.nodes[key].dependencies)
        return seen


def create_app(*, scheduler: Scheduler, token: str) -> FastAPI:
    app = FastAPI()

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.post(
        "/api/v1/submissions",
        response_model=CreateSubmissionResponse,
        dependencies=[Depends(authorize)],
    )
    def create_submission(req: CreateSubmissionRequest) -> CreateSubmissionResponse:
        submission_id = scheduler.create_submission(req)
        scheduler.schedule_queued_nodes()
        return CreateSubmissionResponse(submission_id=submission_id)

    @app.get(
        "/api/v1/submissions/{submission_id}",
        response_model=SubmissionStatusResponse,
        dependencies=[Depends(authorize)],
    )
    def get_submission(submission_id: str) -> SubmissionStatusResponse:
        try:
            return scheduler.status(submission_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown submission") from exc

    @app.get(
        "/api/v1/leases/{lease_id}",
        response_model=LeaseResponse,
        dependencies=[Depends(authorize)],
    )
    def get_lease(lease_id: str) -> LeaseResponse:
        try:
            lease = scheduler.leases[lease_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown lease") from exc
        return LeaseResponse(
            lease_id=lease.lease_id,
            submission_id=lease.submission_id,
            node=lease.node,
        )

    @app.post("/api/v1/leases/{lease_id}/result", dependencies=[Depends(authorize)])
    def post_lease_result(lease_id: str, result: LeaseResult) -> dict[str, bool]:
        try:
            scheduler.handle_lease_result(lease_id, result)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.get(
        "/api/v1/job-groups/{job_group_id}/leases/{array_index}",
        response_model=LeaseResponse,
        dependencies=[Depends(authorize)],
    )
    def get_group_lease(job_group_id: str, array_index: int) -> LeaseResponse:
        try:
            lease_id = scheduler.job_groups[job_group_id][array_index]
            lease = scheduler.leases[lease_id]
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="unknown group lease") from exc
        return LeaseResponse(
            lease_id=lease.lease_id,
            submission_id=lease.submission_id,
            node=lease.node,
        )

    return app


class SchedulerClient:
    def create_submission(self, req: CreateSubmissionRequest) -> str:
        raise NotImplementedError

    def get_submission_status(self, submission_id: str) -> SubmissionStatusResponse:
        raise NotImplementedError

    def get_lease(self, lease_id: str) -> LeaseRecord:
        raise NotImplementedError

    def get_lease_for_array_index(
        self, *, job_group_id: str, array_index: int
    ) -> LeaseRecord:
        raise NotImplementedError

    def post_lease_result(self, *, lease_id: str, result: LeaseResult) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HttpSchedulerClient(SchedulerClient):
    def __init__(self, *, server_url: str, token: str) -> None:
        self._client = httpx.Client(
            base_url=server_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=None,
        )

    def create_submission(self, req: CreateSubmissionRequest) -> str:
        response = self._client.post(
            "/api/v1/submissions", json=req.model_dump(mode="json")
        )
        response.raise_for_status()
        return CreateSubmissionResponse.model_validate(response.json()).submission_id

    def get_submission_status(self, submission_id: str) -> SubmissionStatusResponse:
        response = self._client.get(f"/api/v1/submissions/{submission_id}")
        response.raise_for_status()
        return SubmissionStatusResponse.model_validate(response.json())

    def get_lease(self, lease_id: str) -> LeaseRecord:
        response = self._client.get(f"/api/v1/leases/{lease_id}")
        response.raise_for_status()
        lease = LeaseResponse.model_validate(response.json())
        return LeaseRecord(
            lease_id=lease.lease_id,
            submission_id=lease.submission_id,
            node=lease.node,
        )

    def get_lease_for_array_index(
        self, *, job_group_id: str, array_index: int
    ) -> LeaseRecord:
        response = self._client.get(
            f"/api/v1/job-groups/{job_group_id}/leases/{array_index}"
        )
        response.raise_for_status()
        lease = LeaseResponse.model_validate(response.json())
        return LeaseRecord(
            lease_id=lease.lease_id,
            submission_id=lease.submission_id,
            node=lease.node,
        )

    def post_lease_result(self, *, lease_id: str, result: LeaseResult) -> None:
        response = self._client.post(
            f"/api/v1/leases/{lease_id}/result", json=result.model_dump(mode="json")
        )
        response.raise_for_status()

    def close(self) -> None:
        self._client.close()


class TestSchedulerClient(SchedulerClient):
    def __init__(self, *, app: FastAPI, token: str) -> None:
        self._client = TestClient(
            app,
            headers={"Authorization": f"Bearer {token}"},
        )

    def create_submission(self, req: CreateSubmissionRequest) -> str:
        response = self._client.post(
            "/api/v1/submissions", json=req.model_dump(mode="json")
        )
        response.raise_for_status()
        return CreateSubmissionResponse.model_validate(response.json()).submission_id

    def get_submission_status(self, submission_id: str) -> SubmissionStatusResponse:
        response = self._client.get(f"/api/v1/submissions/{submission_id}")
        response.raise_for_status()
        return SubmissionStatusResponse.model_validate(response.json())

    def get_lease(self, lease_id: str) -> LeaseRecord:
        response = self._client.get(f"/api/v1/leases/{lease_id}")
        response.raise_for_status()
        lease = LeaseResponse.model_validate(response.json())
        return LeaseRecord(
            lease_id=lease.lease_id,
            submission_id=lease.submission_id,
            node=lease.node,
        )

    def get_lease_for_array_index(
        self, *, job_group_id: str, array_index: int
    ) -> LeaseRecord:
        response = self._client.get(
            f"/api/v1/job-groups/{job_group_id}/leases/{array_index}"
        )
        response.raise_for_status()
        lease = LeaseResponse.model_validate(response.json())
        return LeaseRecord(
            lease_id=lease.lease_id,
            submission_id=lease.submission_id,
            node=lease.node,
        )

    def post_lease_result(self, *, lease_id: str, result: LeaseResult) -> None:
        response = self._client.post(
            f"/api/v1/leases/{lease_id}/result", json=result.model_dump(mode="json")
        )
        response.raise_for_status()


def new_token() -> str:
    return secrets.token_urlsafe(32)
