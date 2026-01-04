"""Pydantic models for the Dashboard API."""

from typing import Any

from pydantic import BaseModel

from ...storage import StateAttempt


# Type alias for JSON-serializable data from Pydantic model_dump(mode="json").
# This is the output of serializing Huldra state/metadata models to JSON format.
JsonDict = dict[str, Any]


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    version: str


class ExperimentSummary(BaseModel):
    """Summary of an experiment for list views."""

    namespace: str
    huldra_hash: str
    class_name: str
    result_status: str
    attempt_status: str | None = None
    attempt_number: int | None = None
    updated_at: str | None = None
    started_at: str | None = None
    # Additional fields for filtering
    backend: str | None = None
    hostname: str | None = None
    user: str | None = None


class ExperimentDetail(ExperimentSummary):
    """Detailed experiment information."""

    directory: str
    state: JsonDict
    metadata: JsonDict | None = None
    attempt: StateAttempt | None = None


class ExperimentList(BaseModel):
    """List of experiments with total count."""

    experiments: list[ExperimentSummary]
    total: int


class StatusCount(BaseModel):
    """Count of experiments by status."""

    status: str
    count: int


class DashboardStats(BaseModel):
    """Aggregate dashboard statistics."""

    total: int
    by_result_status: list[StatusCount]
    by_attempt_status: list[StatusCount]
    running_count: int
    queued_count: int
    failed_count: int
    success_count: int


# DAG Models


class DAGExperiment(BaseModel):
    """An experiment instance within a DAG node (grouped by class)."""

    namespace: str
    huldra_hash: str
    result_status: str
    attempt_status: str | None = None


class DAGNode(BaseModel):
    """A node in the experiment DAG representing a class type.

    Multiple experiments of the same class are grouped into one node.
    """

    id: str  # Unique node identifier (class name)
    class_name: str  # Short class name (e.g., "TrainModel")
    full_class_name: str  # Full qualified class name from __class__
    experiments: list[DAGExperiment]  # All experiments of this class
    # Counts by status for quick access
    total_count: int
    success_count: int
    failed_count: int
    running_count: int
    # Parent class for subclass relationships
    parent_class: str | None = None


class DAGEdge(BaseModel):
    """An edge in the experiment DAG representing a dependency."""

    source: str  # Source node id (parent/upstream)
    target: str  # Target node id (child/downstream)
    field_name: str  # The field name that creates this dependency


class ExperimentDAG(BaseModel):
    """Complete DAG structure for visualization."""

    nodes: list[DAGNode]
    edges: list[DAGEdge]
    # Summary statistics
    total_nodes: int
    total_edges: int
    total_experiments: int
