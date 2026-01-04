"""Filesystem scanner for discovering and parsing Huldra experiment state."""

import json
from collections import defaultdict
from pathlib import Path

from ..config import HULDRA_CONFIG
from ..storage import StateAttempt
from ..storage.state import StateManager, _HuldraState
from .api.models import (
    DashboardStats,
    ExperimentDetail,
    ExperimentSummary,
    JsonDict,
    StatusCount,
)


def _parse_namespace_from_path(experiment_dir: Path, root: Path) -> tuple[str, str]:
    """
    Parse namespace and huldra_hash from experiment directory path.

    Example: /data/my_project/pipelines/TrainModel/abc123 -> ("my_project.pipelines.TrainModel", "abc123")
    """
    relative = experiment_dir.relative_to(root)
    parts = relative.parts
    if len(parts) < 2:  # TODO: Maybe this should throw?
        return str(relative), ""
    huldra_hash = parts[-1]
    namespace = ".".join(parts[:-1])
    return namespace, huldra_hash


def _get_class_name(namespace: str) -> str:
    """Extract class name from namespace (last component)."""
    parts = namespace.split(".")
    return parts[-1] if parts else namespace


def _state_to_summary(
    state: _HuldraState, namespace: str, huldra_hash: str
) -> ExperimentSummary:
    """Convert a Huldra state to an experiment summary."""
    attempt = state.attempt
    return ExperimentSummary(
        namespace=namespace,
        huldra_hash=huldra_hash,
        class_name=_get_class_name(namespace),
        result_status=state.result.status,
        attempt_status=attempt.status if attempt else None,
        attempt_number=attempt.number if attempt else None,
        updated_at=state.updated_at,
        started_at=attempt.started_at if attempt else None,
    )


def _state_to_detail(
    state: _HuldraState,
    namespace: str,
    huldra_hash: str,
    directory: Path,
    metadata: JsonDict | None,
) -> ExperimentDetail:
    """Convert a Huldra state to a detailed experiment record."""
    attempt = state.attempt
    attempt_detail = StateAttempt.from_internal(attempt) if attempt else None

    return ExperimentDetail(
        namespace=namespace,
        huldra_hash=huldra_hash,
        class_name=_get_class_name(namespace),
        result_status=state.result.status,
        attempt_status=attempt.status if attempt else None,
        attempt_number=attempt.number if attempt else None,
        updated_at=state.updated_at,
        started_at=attempt.started_at if attempt else None,
        directory=str(directory),
        state=state.model_dump(mode="json"),
        metadata=metadata,
        attempt=attempt_detail,
    )


def _find_experiment_dirs(root: Path) -> list[Path]:
    """Find all directories containing .huldra/state.json files."""
    experiments = []

    # Walk the directory tree looking for .huldra directories
    for huldra_dir in root.rglob(".huldra"):
        if huldra_dir.is_dir():
            state_file = huldra_dir / "state.json"
            if state_file.is_file():
                experiments.append(huldra_dir.parent)

    return experiments


def _read_metadata(directory: Path) -> JsonDict | None:
    """Read metadata.json from an experiment directory."""
    metadata_path = directory / ".huldra" / "metadata.json"
    if not metadata_path.is_file():
        return None
    return json.loads(metadata_path.read_text())


def scan_experiments(
    *,
    result_status: str | None = None,
    attempt_status: str | None = None,
    namespace_prefix: str | None = None,
) -> list[ExperimentSummary]:
    """
    Scan the filesystem for Huldra experiments.

    Args:
        result_status: Filter by result status (absent, incomplete, success, failed)
        attempt_status: Filter by attempt status (queued, running, success, failed, etc.)
        namespace_prefix: Filter by namespace prefix

    Returns:
        List of experiment summaries, sorted by updated_at (newest first)
    """
    experiments: list[ExperimentSummary] = []

    # Scan both data and git roots
    for root in [HULDRA_CONFIG.get_root(False), HULDRA_CONFIG.get_root(True)]:
        if not root.exists():
            continue
        for experiment_dir in _find_experiment_dirs(root):
            state = StateManager.read_state(experiment_dir)
            namespace, huldra_hash = _parse_namespace_from_path(experiment_dir, root)

            summary = _state_to_summary(state, namespace, huldra_hash)

            # Apply filters
            if result_status and summary.result_status != result_status:
                continue
            if attempt_status and summary.attempt_status != attempt_status:
                continue
            if namespace_prefix and not summary.namespace.startswith(namespace_prefix):
                continue

            experiments.append(summary)

    # Sort by updated_at (newest first), with None values at the end
    experiments.sort(
        key=lambda e: (e.updated_at is None, e.updated_at or ""),
        reverse=True,
    )

    return experiments


def get_experiment_detail(namespace: str, huldra_hash: str) -> ExperimentDetail | None:
    """
    Get detailed information about a specific experiment.

    Args:
        namespace: Dot-separated namespace (e.g., "my_project.pipelines.TrainModel")
        huldra_hash: Hash identifying the specific experiment

    Returns:
        Experiment detail or None if not found
    """
    # Convert namespace to path
    namespace_path = Path(*namespace.split("."))

    # Try both data and git roots
    for root in [HULDRA_CONFIG.get_root(False), HULDRA_CONFIG.get_root(True)]:
        experiment_dir = root / namespace_path / huldra_hash
        state_file = experiment_dir / ".huldra" / "state.json"

        if state_file.is_file():
            state = StateManager.read_state(experiment_dir)
            metadata = _read_metadata(experiment_dir)
            return _state_to_detail(
                state, namespace, huldra_hash, experiment_dir, metadata
            )

    return None


def get_stats() -> DashboardStats:
    """
    Get aggregate statistics for the dashboard.

    Returns:
        Dashboard statistics including counts by status
    """
    result_counts: dict[str, int] = defaultdict(int)
    attempt_counts: dict[str, int] = defaultdict(int)
    total = 0
    running = 0
    queued = 0
    failed = 0
    success = 0

    # Scan both data and git roots
    for root in [HULDRA_CONFIG.get_root(False), HULDRA_CONFIG.get_root(True)]:
        if not root.exists():
            continue
        for experiment_dir in _find_experiment_dirs(root):
            state = StateManager.read_state(experiment_dir)
            total += 1

            result_counts[state.result.status] += 1

            if state.result.status == "success":
                success += 1
            elif state.result.status == "failed":
                failed += 1

            attempt = state.attempt
            if attempt:
                attempt_counts[attempt.status] += 1
                if attempt.status == "running":
                    running += 1
                elif attempt.status == "queued":
                    queued += 1

    return DashboardStats(
        total=total,
        by_result_status=[
            StatusCount(status=status, count=count)
            for status, count in sorted(result_counts.items())
        ],
        by_attempt_status=[
            StatusCount(status=status, count=count)
            for status, count in sorted(attempt_counts.items())
        ],
        running_count=running,
        queued_count=queued,
        failed_count=failed,
        success_count=success,
    )
