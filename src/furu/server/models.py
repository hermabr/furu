from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from furu.graph import ArtifactNode, GraphEdge, GraphFragment, NodeKey
from furu.metadata import ArtifactSpec


class NodeState(str, Enum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubmissionState(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    node_key: NodeKey
    error_type: str
    error_message: str
    traceback: str


class CreateSubmissionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    graph: GraphFragment
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
    node_key: NodeKey
    artifact: ArtifactSpec


class CompleteLeaseRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    node_key: NodeKey


class DependencyLeaseRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    blocked: NodeKey
    call_kind: Literal["load_or_create", "try_load"]
    dependencies: tuple[NodeKey, ...]
    graph_fragment: GraphFragment


class FailLeaseRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    node_key: NodeKey
    error_type: str
    error_message: str
    traceback: str


class ReserveLeasesRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    max_count: int


class ReserveLeasesResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    leases: tuple[LeaseResponse, ...]


class CreateJobGroupRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    lease_ids: tuple[str, ...]


class CreateJobGroupResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    job_group_id: str


class NodeStatus(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    key: NodeKey
    artifact: ArtifactSpec
    state: NodeState
    active_lease_id: str | None = None
    failure: FailureRecord | None = None


class SubmissionStatus(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    state: SubmissionState
    roots: tuple[NodeKey, ...]
    input_order: tuple[NodeKey, ...]
    single_input: bool
    failure_summary: str | None = None


class SubmissionGraphView(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    nodes: tuple[NodeStatus, ...]
    edges: tuple[GraphEdge, ...]
    roots: tuple[NodeKey, ...]
    input_order: tuple[NodeKey, ...]
    single_input: bool


def lease_response_from_node(
    *,
    lease_id: str,
    submission_id: str,
    node: ArtifactNode,
) -> LeaseResponse:
    return LeaseResponse(
        lease_id=lease_id,
        submission_id=submission_id,
        node_key=node.key,
        artifact=node.artifact,
    )
