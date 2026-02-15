"""API route definitions for the Furu Dashboard."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from .. import __version__
from ..scanner import (
    scan_experiments,
    get_experiment_detail,
    get_stats,
    get_experiment_dag,
    get_experiment_relationships,
)
from .models import (
    DashboardStats,
    ExperimentDAG,
    ExperimentDetail,
    ExperimentList,
    ExperimentSearchRequest,
    ExperimentRelationships,
    HealthCheck,
)

router = APIRouter(prefix="/api", tags=["api"])

SchemaFilter = Literal["current", "stale", "any"]


@router.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """Health check endpoint."""
    return HealthCheck(status="healthy", version=__version__)


@router.get("/experiments", response_model=ExperimentList)
async def list_experiments(
    result_status: str | None = Query(None, description="Filter by result status"),
    attempt_status: str | None = Query(None, description="Filter by attempt status"),
    namespace: str | None = Query(None, description="Filter by namespace prefix"),
    backend: str | None = Query(
        None, description="Filter by backend (local, submitit)"
    ),
    hostname: str | None = Query(None, description="Filter by hostname"),
    user: str | None = Query(None, description="Filter by user"),
    started_after: str | None = Query(
        None, description="Filter experiments started after this ISO datetime"
    ),
    started_before: str | None = Query(
        None, description="Filter experiments started before this ISO datetime"
    ),
    updated_after: str | None = Query(
        None, description="Filter experiments updated after this ISO datetime"
    ),
    updated_before: str | None = Query(
        None, description="Filter experiments updated before this ISO datetime"
    ),
    config_filter: str | None = Query(
        None, description="Filter by config field (format: field.path=value)"
    ),
    migration_kind: str | None = Query(None, description="Filter by migration kind"),
    migration_policy: str | None = Query(
        None, description="Filter by migration policy"
    ),
    schema: SchemaFilter = Query(
        "current", description="Filter by schema status (current, stale, any)"
    ),
    view: str = Query("resolved", description="View mode: resolved or original"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> ExperimentList:
    """List all experiments with optional filtering."""
    experiments = scan_experiments(
        result_status=result_status,
        attempt_status=attempt_status,
        namespace_prefix=namespace,
        backend=backend,
        hostname=hostname,
        user=user,
        started_after=started_after,
        started_before=started_before,
        updated_after=updated_after,
        updated_before=updated_before,
        config_filter=config_filter,
        migration_kind=migration_kind,
        migration_policy=migration_policy,
        schema=schema,
        view=view,
    )

    # Apply pagination
    total = len(experiments)
    experiments = experiments[offset : offset + limit]

    return ExperimentList(experiments=experiments, total=total)


@router.post("/experiments/search", response_model=ExperimentList)
async def search_experiments(request: ExperimentSearchRequest) -> ExperimentList:
    """Search experiments using a structured query AST."""
    try:
        experiments = scan_experiments(
            query=request.query,
            schema=request.schema_filter,
            view=request.view,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    total = len(experiments)
    experiments = experiments[request.offset : request.offset + request.limit]

    return ExperimentList(experiments=experiments, total=total)


@router.get(
    "/experiments/{namespace:path}/{furu_hash}/relationships",
    response_model=ExperimentRelationships,
)
async def get_experiment_relationships_route(
    namespace: str,
    furu_hash: str,
    view: str = Query("resolved", description="View mode: resolved or original"),
) -> ExperimentRelationships:
    """Get parent and child relationships for a specific experiment."""
    relationships = get_experiment_relationships(namespace, furu_hash, view=view)
    if relationships is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return relationships


@router.get(
    "/experiments/{namespace:path}/{furu_hash}", response_model=ExperimentDetail
)
async def get_experiment(
    namespace: str,
    furu_hash: str,
    view: str = Query("resolved", description="View mode: resolved or original"),
) -> ExperimentDetail:
    """Get detailed information about a specific experiment."""
    experiment = get_experiment_detail(namespace, furu_hash, view=view)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.get("/stats", response_model=DashboardStats)
async def dashboard_stats() -> DashboardStats:
    """Get aggregate statistics for the dashboard."""
    return get_stats()


@router.get("/dag", response_model=ExperimentDAG)
async def experiment_dag() -> ExperimentDAG:
    """Get the experiment dependency DAG.

    Returns a graph structure where:
    - Nodes represent experiment classes (e.g., TrainModel, PrepareDataset)
    - Multiple experiments of the same class are grouped into a single node
    - Edges represent dependencies between classes
    """
    return get_experiment_dag()
