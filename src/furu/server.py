from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from furu.core import Furu
from furu.graph import ArtifactNode, DiscoveredGraph, NodeKey, discover_missing_closure
from furu.metadata import ArtifactSpec
from furu.migration import result_dir_for_loading
from furu.submission import SubmissionStatus


class NodeRunState(Enum):
    QUEUED = "queued"
    RUNNING = "running"


@dataclass
class FailureRecord:
    error_type: str
    error_message: str
    traceback: str

    @property
    def message(self) -> str:
        return f"{self.error_type}: {self.error_message}"


@dataclass
class NodeRecord:
    key: NodeKey
    artifact: ArtifactSpec
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


class LeaseRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

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
    def _coerce_tuple_fields(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class CreateSubmissionResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    submission_id: str


class LeaseResponse(LeaseRecord):
    pass


class Launcher(Protocol):
    def capacity(self) -> int: ...

    def launch(self, leases: Sequence[LeaseRecord]) -> None: ...


class Scheduler:
    def __init__(self, *, launcher: Launcher) -> None:
        self.lock = threading.RLock()
        self.submissions: dict[str, SubmissionRecord] = {}
        self.nodes: dict[NodeKey, NodeRecord] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self.job_groups: dict[str, tuple[str, ...]] = {}
        self.launcher = launcher

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

    def create_job_group(self, leases: Sequence[LeaseRecord]) -> str:
        with self.lock:
            job_group_id = uuid.uuid4().hex
            self.job_groups[job_group_id] = tuple(lease.lease_id for lease in leases)
            return job_group_id

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
            try:
                lease_id = lease_ids[array_index]
            except IndexError as exc:
                raise KeyError(
                    f"array index {array_index} not in job group {job_group_id}"
                ) from exc
            return self.validate_lease(lease_id)

    def get_submission_status(self, submission_id: str) -> SubmissionStatus:
        with self.lock:
            submission = self.submissions[submission_id]
            closure = self._submission_nodes(submission)
            nodes = [self.nodes[key] for key in closure]
            failures = [node.failure for node in nodes if node.failure is not None]
            done_nodes = sum(is_done(node) for node in nodes)
            queued_nodes = sum(node.state == NodeRunState.QUEUED for node in nodes)
            running_nodes = sum(
                node.state == NodeRunState.RUNNING or node.active_lease_id is not None
                for node in nodes
            )

            if failures:
                status: Literal["running", "done", "failed"] = "failed"
                failure_message = failures[0].message
            elif all(is_done(self.nodes[root]) for root in submission.roots):
                status = "done"
                failure_message = None
            else:
                status = "running"
                failure_message = None

            return SubmissionStatus(
                submission_id=submission_id,
                status=status,
                failure_message=failure_message,
                total_nodes=len(nodes),
                done_nodes=done_nodes,
                queued_nodes=queued_nodes,
                running_nodes=running_nodes,
            )

    def merge_graph(self, graph: DiscoveredGraph) -> None:
        for artifact_node in graph.nodes:
            self.nodes.setdefault(
                artifact_node.key,
                NodeRecord(
                    key=artifact_node.key,
                    artifact=artifact_node.artifact,
                ),
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
            if is_done(node):
                node.state = None
                node.active_lease_id = None
                continue

            if node.failure is not None:
                node.state = None
                node.active_lease_id = None
                continue

            if self.has_failed_dependency(node):
                node.state = None
                continue

            if node.active_lease_id is not None:
                node.state = NodeRunState.RUNNING
                continue

            if self.deps_done(node):
                node.state = NodeRunState.QUEUED
            else:
                node.state = None

    def schedule_queued_nodes(self) -> None:
        with self.lock:
            self.recompute_queue_state()
            capacity = self.launcher.capacity() - self.running_count()
            if capacity <= 0:
                return

            queued_nodes = [
                node
                for node in self.nodes.values()
                if node.state == NodeRunState.QUEUED
            ]
            leases = [self.reserve_lease(node) for node in queued_nodes[:capacity]]

        if leases:
            self.launcher.launch(leases)

    def reserve_lease(self, node: NodeRecord) -> LeaseRecord:
        if node.state != NodeRunState.QUEUED or node.active_lease_id is not None:
            raise RuntimeError("node is not available for leasing")

        lease_id = uuid.uuid4().hex
        lease = LeaseRecord(
            lease_id=lease_id,
            submission_id=self.submission_for_node(node.key),
            node=ArtifactNode(
                key=node.key,
                artifact=node.artifact,
            ),
        )
        self.leases[lease_id] = lease
        node.active_lease_id = lease_id
        node.state = NodeRunState.RUNNING
        return lease

    def running_count(self) -> int:
        return sum(
            node.state == NodeRunState.RUNNING or node.active_lease_id is not None
            for node in self.nodes.values()
        )

    def handle_lease_result(self, lease_id: str, result: LeaseResult) -> None:
        match result:
            case LeaseDone():
                self.handle_lease_result_done(lease_id, result)
            case LeaseDependencyNotReady():
                self.handle_lease_result_dependency(lease_id, result)
            case LeaseFailed():
                self.handle_lease_result_failed(lease_id, result)

    def handle_lease_result_done(self, lease_id: str, result: LeaseDone) -> None:
        with self.lock:
            self.validate_lease(lease_id)
            node = self.nodes[result.node_key]
            self.clear_lease(node, lease_id)

            if not is_done(node):
                node.failure = FailureRecord(
                    error_type="MissingResult",
                    error_message="Worker reported success but result does not exist",
                    traceback="",
                )

            self.recompute_queue_state()

        self.schedule_queued_nodes()

    def handle_lease_result_dependency(
        self,
        lease_id: str,
        result: LeaseDependencyNotReady,
    ) -> None:
        with self.lock:
            self.validate_lease(lease_id)
            blocked = self.nodes[result.blocked]
            self.clear_lease(blocked, lease_id)
            blocked.state = None

        dependency_objects = [
            Furu.from_artifact(dep.artifact) for dep in result.dependencies
        ]
        graph = discover_missing_closure(dependency_objects)

        with self.lock:
            self.merge_graph(graph)
            for dep in result.dependencies:
                self.add_edge(
                    dependency=dep.key,
                    dependent=result.blocked,
                )
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

    def validate_lease(self, lease_id: str) -> LeaseRecord:
        try:
            return self.leases[lease_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown lease: {lease_id}") from exc

    def clear_lease(self, node: NodeRecord, lease_id: str) -> None:
        stale_lease = self.leases.get(lease_id)
        if (
            node.active_lease_id is None
            and stale_lease is not None
            and stale_lease.node.key == node.key
        ):
            node.state = None
            self.leases.pop(lease_id, None)
            return

        if node.active_lease_id != lease_id:
            raise RuntimeError("lease does not match active node lease")
        node.active_lease_id = None
        node.state = None
        self.leases.pop(lease_id, None)

    def has_failed_dependency(self, node: NodeRecord) -> bool:
        return any(self.nodes[dep].failure is not None for dep in node.dependencies)

    def deps_done(self, node: NodeRecord) -> bool:
        return all(is_done(self.nodes[dep]) for dep in node.dependencies)

    def submission_for_node(self, key: NodeKey) -> str:
        for submission in self.submissions.values():
            if key in self._submission_nodes(submission):
                return submission.id
        if self.submissions:
            return next(iter(self.submissions))
        raise RuntimeError("cannot reserve a lease without a submission")

    def _submission_nodes(self, submission: SubmissionRecord) -> set[NodeKey]:
        seen: set[NodeKey] = set()
        stack = list(submission.roots)
        while stack:
            key = stack.pop()
            if key in seen or key not in self.nodes:
                continue
            seen.add(key)
            stack.extend(self.nodes[key].dependencies)
        return seen


def node_object(node: NodeRecord) -> Furu[Any]:
    return Furu.from_artifact(node.artifact)


def is_done(node: NodeRecord) -> bool:
    return result_dir_for_loading(node_object(node)) is not None


def create_app(*, scheduler: Scheduler, token: str) -> FastAPI:
    app = FastAPI()

    def require_auth(request: Request) -> None:
        expected = f"Bearer {token}"
        if request.headers.get("authorization") != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.post("/api/v1/submissions")
    def create_submission(
        req: CreateSubmissionRequest,
        request: Request,
    ) -> CreateSubmissionResponse:
        require_auth(request)
        submission_id = scheduler.create_submission(req)
        scheduler.schedule_queued_nodes()
        return CreateSubmissionResponse(submission_id=submission_id)

    @app.get("/api/v1/submissions/{submission_id}")
    def get_submission_status(
        submission_id: str,
        request: Request,
    ) -> SubmissionStatus:
        require_auth(request)
        try:
            return scheduler.get_submission_status(submission_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown submission") from exc

    @app.get("/api/v1/leases/{lease_id}")
    def get_lease(lease_id: str, request: Request) -> LeaseResponse:
        require_auth(request)
        try:
            lease = scheduler.validate_lease(lease_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return LeaseResponse.model_validate(lease.model_dump())

    @app.post("/api/v1/leases/{lease_id}/result")
    def report_lease_result(
        lease_id: str,
        result: LeaseResult,
        request: Request,
    ) -> dict[str, bool]:
        require_auth(request)
        scheduler.handle_lease_result(lease_id, result)
        return {"ok": True}

    @app.get("/api/v1/job-groups/{job_group_id}/leases/{array_index}")
    def get_lease_for_array_index(
        job_group_id: str,
        array_index: int,
        request: Request,
    ) -> LeaseResponse:
        require_auth(request)
        try:
            lease = scheduler.get_lease_for_array_index(
                job_group_id=job_group_id,
                array_index=array_index,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return LeaseResponse.model_validate(lease.model_dump())

    return app


class SchedulerClient:
    def __init__(self, http: Any, *, token: str) -> None:
        self._http = http
        self._headers = {"Authorization": f"Bearer {token}"}

    @classmethod
    def for_url(cls, server_url: str, *, token: str) -> SchedulerClient:
        return cls(httpx.Client(base_url=server_url, timeout=None), token=token)

    def create_submission(
        self,
        *,
        graph: DiscoveredGraph,
        roots: tuple[NodeKey, ...],
        input_order: tuple[NodeKey, ...],
        single_input: bool,
    ) -> str:
        req = CreateSubmissionRequest(
            graph=graph,
            roots=roots,
            input_order=input_order,
            single_input=single_input,
        )
        response = self._http.post(
            "/api/v1/submissions",
            json=req.model_dump(mode="json"),
            headers=self._headers,
        )
        response.raise_for_status()
        return CreateSubmissionResponse.model_validate(response.json()).submission_id

    def get_submission_status(self, submission_id: str) -> SubmissionStatus:
        response = self._http.get(
            f"/api/v1/submissions/{submission_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        return SubmissionStatus.model_validate(response.json())

    def get_lease(self, lease_id: str) -> LeaseRecord:
        response = self._http.get(
            f"/api/v1/leases/{lease_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        return LeaseRecord.model_validate(response.json())

    def get_lease_for_array_index(
        self,
        *,
        job_group_id: str,
        array_index: int,
    ) -> LeaseRecord:
        response = self._http.get(
            f"/api/v1/job-groups/{job_group_id}/leases/{array_index}",
            headers=self._headers,
        )
        response.raise_for_status()
        return LeaseRecord.model_validate(response.json())

    def post_lease_result(self, *, lease_id: str, result: LeaseResult) -> None:
        response = self._http.post(
            f"/api/v1/leases/{lease_id}/result",
            json=result.model_dump(mode="json"),
            headers=self._headers,
        )
        response.raise_for_status()

    def close(self) -> None:
        close = getattr(self._http, "close", None)
        if close is not None:
            close()
