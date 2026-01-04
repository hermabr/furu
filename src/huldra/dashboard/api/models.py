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
