"""Pydantic models for the Dashboard API."""

from typing import Any

from pydantic import BaseModel

from ...storage import StateAttempt


# Type alias for JSON-serializable data from Pydantic model_dump(mode="json").
# This is the output of serializing Furu state/metadata models to JSON format.
JsonDict = dict[str, Any]


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    version: str


class AliasInfo(BaseModel):
    """Alias reference information for an experiment."""

    namespace: str
    furu_hash: str
    migrated_at: str | None = None
    overwritten_at: str | None = None
    origin: str | None = None
    note: str | None = None


class ExperimentSummary(BaseModel):
    """Summary of an experiment for list views."""

    namespace: str
    furu_hash: str
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
    # Migration metadata
    migration_kind: str | None = None
    migration_policy: str | None = None
    migrated_at: str | None = None
    overwritten_at: str | None = None
    migration_origin: str | None = None
    migration_note: str | None = None
    from_namespace: str | None = None
    from_hash: str | None = None
    to_namespace: str | None = None
    to_hash: str | None = None
    original_result_status: str | None = None
    original_namespace: str | None = None
    original_hash: str | None = None
    schema_key: list[str] | None = None
    current_schema_key: list[str] | None = None
    is_stale: bool | None = None
    is_alias: bool | None = None
    aliases: list[AliasInfo] | None = None


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
    furu_hash: str
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


# Parent/Child relationship models


class ParentExperiment(BaseModel):
    """A parent experiment that this experiment depends on."""

    field_name: str  # The field name that references this parent
    class_name: str  # Short class name
    full_class_name: str  # Full qualified class name
    namespace: str | None = None  # Namespace if the experiment exists
    furu_hash: str | None = None  # Hash if the experiment exists
    result_status: str | None = None  # Status if the experiment exists
    config: dict[str, Any] | None = None  # Parent's config for identification


class ChildExperiment(BaseModel):
    """A child experiment that depends on this experiment."""

    field_name: str  # The field name through which this experiment is referenced
    class_name: str  # Short class name of the child
    full_class_name: str  # Full qualified class name of the child
    namespace: str  # Namespace of the child experiment
    furu_hash: str  # Hash of the child experiment
    result_status: str  # Status of the child experiment


class ExperimentRelationships(BaseModel):
    """Parent and child relationships for an experiment."""

    parents: list[ParentExperiment]
    children: list[ChildExperiment]
