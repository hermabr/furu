"""Filesystem scanner for discovering and parsing Gren experiment state."""

import datetime as _dt
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from ..config import GREN_CONFIG
from ..storage import MetadataManager, MigrationManager, MigrationRecord, StateAttempt
from ..storage.state import StateManager, _GrenState
from .api.models import (
    ChildExperiment,
    DAGEdge,
    DAGExperiment,
    DAGNode,
    DashboardStats,
    ExperimentDAG,
    ExperimentDetail,
    ExperimentRelationships,
    ExperimentSummary,
    JsonDict,
    ParentExperiment,
    StatusCount,
)


def _iter_roots() -> Iterator[Path]:
    """Iterate over all existing Gren storage roots."""
    for version_controlled in (False, True):
        root = GREN_CONFIG.get_root(version_controlled)
        if root.exists():
            yield root


def _parse_namespace_from_path(experiment_dir: Path, root: Path) -> tuple[str, str]:
    """
    Parse namespace and gren_hash from experiment directory path.

    Example: /data/my_project/pipelines/TrainModel/abc123 -> ("my_project.pipelines.TrainModel", "abc123")
    """
    relative = experiment_dir.relative_to(root)
    parts = relative.parts
    if len(parts) < 2:  # TODO: Maybe this should throw?
        return str(relative), ""
    gren_hash = parts[-1]
    namespace = ".".join(parts[:-1])
    return namespace, gren_hash


def _get_class_name(namespace: str) -> str:
    """Extract class name from namespace (last component)."""
    parts = namespace.split(".")
    return parts[-1] if parts else namespace


def _state_to_summary(
    state: _GrenState,
    namespace: str,
    gren_hash: str,
    migration: MigrationRecord | None = None,
    original_status: str | None = None,
) -> ExperimentSummary:
    """Convert a Gren state to an experiment summary."""
    attempt = state.attempt
    return ExperimentSummary(
        namespace=namespace,
        gren_hash=gren_hash,
        class_name=_get_class_name(namespace),
        result_status=state.result.status,
        attempt_status=attempt.status if attempt else None,
        attempt_number=attempt.number if attempt else None,
        updated_at=state.updated_at,
        started_at=attempt.started_at if attempt else None,
        # Additional fields for filtering
        backend=attempt.backend if attempt else None,
        hostname=attempt.owner.hostname if attempt else None,
        user=attempt.owner.user if attempt else None,
        migration_kind=migration.kind if migration else None,
        migration_policy=migration.policy if migration else None,
        migrated_at=migration.migrated_at if migration else None,
        overwritten_at=migration.overwritten_at if migration else None,
        migration_origin=migration.origin if migration else None,
        migration_note=migration.note if migration else None,
        from_namespace=migration.from_namespace if migration else None,
        from_hash=migration.from_hash if migration else None,
        to_namespace=migration.to_namespace if migration else None,
        to_hash=migration.to_hash if migration else None,
        original_result_status=original_status,
    )


def _override_summary_attempts(
    summary: ExperimentSummary, state: _GrenState
) -> ExperimentSummary:
    attempt = state.attempt
    return summary.model_copy(
        update={
            "attempt_status": attempt.status if attempt else None,
            "attempt_number": attempt.number if attempt else None,
            "started_at": attempt.started_at if attempt else None,
            "backend": attempt.backend if attempt else None,
            "hostname": attempt.owner.hostname if attempt else None,
            "user": attempt.owner.user if attempt else None,
        }
    )


def _state_to_detail(
    state: _GrenState,
    namespace: str,
    gren_hash: str,
    directory: Path,
    metadata: JsonDict | None,
    migration: MigrationRecord | None = None,
    original_status: str | None = None,
    original_namespace: str | None = None,
    original_hash: str | None = None,
) -> ExperimentDetail:
    """Convert a Gren state to a detailed experiment record."""
    attempt = state.attempt
    attempt_detail = StateAttempt.from_internal(attempt) if attempt else None

    return ExperimentDetail(
        namespace=namespace,
        gren_hash=gren_hash,
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
        migration_kind=migration.kind if migration else None,
        migration_policy=migration.policy if migration else None,
        migrated_at=migration.migrated_at if migration else None,
        overwritten_at=migration.overwritten_at if migration else None,
        migration_origin=migration.origin if migration else None,
        migration_note=migration.note if migration else None,
        from_namespace=migration.from_namespace if migration else None,
        from_hash=migration.from_hash if migration else None,
        to_namespace=migration.to_namespace if migration else None,
        to_hash=migration.to_hash if migration else None,
        original_result_status=original_status,
        original_namespace=original_namespace,
        original_hash=original_hash,
    )


def _find_experiment_dirs(root: Path) -> list[Path]:
    """Find all directories containing .gren/state.json files."""
    experiments = []

    # Walk the directory tree looking for .gren directories
    for gren_dir in root.rglob(StateManager.INTERNAL_DIR):
        if gren_dir.is_dir():
            state_file = gren_dir / StateManager.STATE_FILE
            if state_file.is_file():
                experiments.append(gren_dir.parent)

    return experiments


def _parse_datetime(value: str | None) -> _dt.datetime | None:
    """Parse ISO datetime string to datetime object."""
    if not value:
        return None
    dt = _dt.datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _get_nested_value(data: dict, path: str) -> str | int | float | bool | None:
    """
    Get a nested value from a dict using dot notation.

    Example: _get_nested_value({"a": {"b": 1}}, "a.b") -> 1
    """
    keys = path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None
        current = current[key]
    # Only return primitive values that can be compared as strings
    if isinstance(current, (str, int, float, bool)):
        return current
    return None


def scan_experiments(
    *,
    result_status: str | None = None,
    attempt_status: str | None = None,
    namespace_prefix: str | None = None,
    backend: str | None = None,
    hostname: str | None = None,
    user: str | None = None,
    started_after: str | None = None,
    started_before: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    config_filter: str | None = None,
    view: str = "resolved",
) -> list[ExperimentSummary]:
    """
    Scan the filesystem for Gren experiments.

    Args:
        result_status: Filter by result status (absent, incomplete, success, failed)
        attempt_status: Filter by attempt status (queued, running, success, failed, etc.)
        namespace_prefix: Filter by namespace prefix
        backend: Filter by backend (local, submitit)
        hostname: Filter by hostname
        user: Filter by user who ran the experiment
        started_after: Filter experiments started after this ISO datetime
        started_before: Filter experiments started before this ISO datetime
        updated_after: Filter experiments updated after this ISO datetime
        updated_before: Filter experiments updated before this ISO datetime
        config_filter: Filter by config field in format "field.path=value"
        view: "resolved" uses alias metadata; "original" uses original metadata/state.

    Returns:
        List of experiment summaries, sorted by updated_at (newest first)
    """
    experiments: list[ExperimentSummary] = []

    # Parse datetime filters
    started_after_dt = _parse_datetime(started_after)
    started_before_dt = _parse_datetime(started_before)
    updated_after_dt = _parse_datetime(updated_after)
    updated_before_dt = _parse_datetime(updated_before)

    # Parse config filter (format: "field.path=value")
    config_field: str | None = None
    config_value: str | None = None
    if config_filter and "=" in config_filter:
        config_field, config_value = config_filter.split("=", 1)

    for root in _iter_roots():
        for experiment_dir in _find_experiment_dirs(root):
            state = StateManager.read_state(experiment_dir)
            namespace, gren_hash = _parse_namespace_from_path(experiment_dir, root)
            migration = MigrationManager.read_migration(experiment_dir)
            original_status: str | None = None
            original_state: _GrenState | None = None
            metadata_dir = experiment_dir
            alias_active = False

            if migration is not None and migration.kind == "alias":
                original_dir = MigrationManager.resolve_dir(migration, target="from")
                original_state = StateManager.read_state(original_dir)
                original_status = original_state.result.status
                alias_active = (
                    migration.overwritten_at is None
                    and state.result.status == "migrated"
                    and original_status == "success"
                )
                if view == "original":
                    state = original_state
                    namespace = migration.from_namespace
                    gren_hash = migration.from_hash
                    metadata_dir = original_dir
                elif alias_active:
                    metadata_dir = original_dir

            summary = _state_to_summary(
                state,
                namespace,
                gren_hash,
                migration=migration,
                original_status=original_status,
            )

            if (
                migration is not None
                and migration.kind == "alias"
                and view == "resolved"
                and alias_active
                and original_state is not None
            ):
                summary = _override_summary_attempts(summary, original_state)

            filter_updated_at = summary.updated_at
            if (
                migration is not None
                and migration.kind == "alias"
                and view == "resolved"
                and alias_active
                and original_state is not None
            ):
                filter_updated_at = original_state.updated_at

            # Apply filters
            if result_status and summary.result_status != result_status:
                continue
            if attempt_status and summary.attempt_status != attempt_status:
                continue
            if namespace_prefix and not summary.namespace.startswith(namespace_prefix):
                continue
            if backend and summary.backend != backend:
                continue
            if hostname and summary.hostname != hostname:
                continue
            if user and summary.user != user:
                continue

            # Date filters
            if started_after_dt or started_before_dt:
                started_dt = _parse_datetime(summary.started_at)
                if started_dt:
                    if started_after_dt and started_dt < started_after_dt:
                        continue
                    if started_before_dt and started_dt > started_before_dt:
                        continue
                elif started_after_dt or started_before_dt:
                    # No started_at but we're filtering by it - exclude
                    continue

            if updated_after_dt or updated_before_dt:
                updated_dt = _parse_datetime(filter_updated_at)
                if updated_dt:
                    if updated_after_dt and updated_dt < updated_after_dt:
                        continue
                    if updated_before_dt and updated_dt > updated_before_dt:
                        continue
                elif updated_after_dt or updated_before_dt:
                    # No updated_at but we're filtering by it - exclude
                    continue

            # Config field filter - requires reading metadata
            if config_field and config_value is not None:
                metadata = MetadataManager.read_metadata_raw(metadata_dir)
                if metadata:
                    gren_obj = metadata.get("gren_obj")
                    if isinstance(gren_obj, dict):
                        actual_value = _get_nested_value(gren_obj, config_field)
                        if str(actual_value) != config_value:
                            continue
                    else:
                        continue
                else:
                    continue

            experiments.append(summary)

    # Sort by updated_at (newest first), with None values at the end
    experiments.sort(
        key=lambda e: (e.updated_at is None, e.updated_at or ""),
        reverse=True,
    )

    return experiments


def get_experiment_detail(
    namespace: str,
    gren_hash: str,
    *,
    view: str = "resolved",
) -> ExperimentDetail | None:
    """
    Get detailed information about a specific experiment.

    Args:
        namespace: Dot-separated namespace (e.g., "my_project.pipelines.TrainModel")
        gren_hash: Hash identifying the specific experiment
        view: "resolved" uses alias metadata; "original" uses original metadata/state.

    Returns:
        Experiment detail or None if not found
    """
    # Convert namespace to path
    namespace_path = Path(*namespace.split("."))

    for root in _iter_roots():
        experiment_dir = root / namespace_path / gren_hash
        state_path = StateManager.get_state_path(experiment_dir)

        if state_path.is_file():
            state = StateManager.read_state(experiment_dir)
            metadata = MetadataManager.read_metadata_raw(experiment_dir)
            migration = MigrationManager.read_migration(experiment_dir)
            original_status: str | None = None
            original_namespace: str | None = None
            original_hash: str | None = None

            if migration is not None and migration.kind == "alias":
                original_dir = MigrationManager.resolve_dir(migration, target="from")
                original_state = StateManager.read_state(original_dir)
                original_status = original_state.result.status
                original_namespace = migration.from_namespace
                original_hash = migration.from_hash
                if view == "original":
                    state = original_state
                    metadata = MetadataManager.read_metadata_raw(original_dir)
                    experiment_dir = original_dir
                    namespace = original_namespace
                    gren_hash = original_hash

            return _state_to_detail(
                state,
                namespace,
                gren_hash,
                experiment_dir,
                metadata,
                migration=migration,
                original_status=original_status,
                original_namespace=original_namespace,
                original_hash=original_hash,
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

    for root in _iter_roots():
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


def _extract_dependencies_from_gren_obj(
    gren_obj: dict[str, object],
) -> list[tuple[str, str]]:
    """
    Extract dependency class names from a serialized gren object.

    Looks for nested objects with __class__ markers, which indicate Gren dependencies.

    Args:
        gren_obj: The serialized gren object (from metadata.gren_obj)

    Returns:
        List of (field_name, dependency_class_name) tuples
    """
    dependencies: list[tuple[str, str]] = []

    for key, value in gren_obj.items():
        if key == "__class__":
            continue
        if isinstance(value, dict):
            nested_obj = cast(dict[str, object], value)
            dep_class_value = nested_obj.get("__class__")
            if dep_class_value is not None:
                # This is a nested Gren object (dependency)
                dependencies.append((key, str(dep_class_value)))

    return dependencies


def _get_class_hierarchy(full_class_name: str) -> str | None:
    """
    Try to determine the parent class from the full class name.

    This is a heuristic - we look at class naming patterns.
    In the future, this could be enhanced to read actual class hierarchies.
    """
    # For now, we don't have access to actual class hierarchies at runtime
    # This would require importing the classes or storing hierarchy info in metadata
    return None


def get_experiment_dag() -> ExperimentDAG:
    """
    Build a DAG of all experiments based on their dependencies.

    The DAG is organized by class types:
    - Each node represents a class (e.g., TrainModel)
    - Experiments of the same class are grouped into the same node
    - Edges represent dependencies between classes (field references)

    Returns:
        ExperimentDAG with nodes and edges for visualization
    """
    # Collect all experiments with their metadata
    experiments_by_class: dict[str, list[tuple[str, str, str, str | None]]] = (
        defaultdict(list)
    )
    # Maps full class name -> (short name, experiments)
    class_info: dict[str, str] = {}  # full_class_name -> short_class_name
    # Collect all edges (deduped by class pair)
    edge_set: set[tuple[str, str, str]] = set()  # (source_class, target_class, field)

    for root in _iter_roots():
        for experiment_dir in _find_experiment_dirs(root):
            state = StateManager.read_state(experiment_dir)
            namespace, gren_hash = _parse_namespace_from_path(experiment_dir, root)
            metadata = MetadataManager.read_metadata_raw(experiment_dir)

            if not metadata:
                continue

            migration = MigrationManager.read_migration(experiment_dir)
            if migration is not None and migration.kind == "alias":
                continue

            gren_obj = metadata.get("gren_obj")
            if not isinstance(gren_obj, dict):
                continue

            full_class_name = gren_obj.get("__class__")
            if not isinstance(full_class_name, str):
                continue

            # Extract short class name
            short_class_name = full_class_name.split(".")[-1]
            class_info[full_class_name] = short_class_name

            # Get attempt status
            attempt_status = state.attempt.status if state.attempt else None

            # Store experiment info
            experiments_by_class[full_class_name].append(
                (namespace, gren_hash, state.result.status, attempt_status)
            )

            # Extract dependencies and create edges
            dependencies = _extract_dependencies_from_gren_obj(gren_obj)
            for field_name, dep_class in dependencies:
                # Edge goes from dependency (source/upstream) to this class (target/downstream)
                edge_set.add((dep_class, full_class_name, field_name))
                # Also make sure the dependency class is in our class_info
                if dep_class not in class_info:
                    class_info[dep_class] = dep_class.split(".")[-1]

    # Build nodes
    nodes: list[DAGNode] = []
    for full_class_name, short_class_name in class_info.items():
        experiments = experiments_by_class.get(full_class_name, [])

        # Count statuses
        success_count = sum(1 for _, _, rs, _ in experiments if rs == "success")
        failed_count = sum(1 for _, _, rs, _ in experiments if rs == "failed")
        running_count = sum(
            1 for _, _, _, attempt_status in experiments if attempt_status == "running"
        )

        node = DAGNode(
            id=full_class_name,
            class_name=short_class_name,
            full_class_name=full_class_name,
            experiments=[
                DAGExperiment(
                    namespace=ns,
                    gren_hash=h,
                    result_status=rs,
                    attempt_status=attempt_status,
                )
                for ns, h, rs, attempt_status in experiments
            ],
            total_count=len(experiments),
            success_count=success_count,
            failed_count=failed_count,
            running_count=running_count,
            parent_class=_get_class_hierarchy(full_class_name),
        )
        nodes.append(node)

    # Build edges
    edges: list[DAGEdge] = [
        DAGEdge(source=source, target=target, field_name=field)
        for source, target, field in edge_set
    ]

    # Sort nodes by class name for consistent ordering
    nodes.sort(key=lambda n: n.class_name)
    edges.sort(key=lambda e: (e.source, e.target))

    return ExperimentDAG(
        nodes=nodes,
        edges=edges,
        total_nodes=len(nodes),
        total_edges=len(edges),
        total_experiments=sum(node.total_count for node in nodes),
    )


def _find_experiment_by_gren_obj(
    gren_obj: dict[str, object],
) -> tuple[str, str, str] | None:
    """
    Find an experiment that matches the given gren_obj serialization.

    Args:
        gren_obj: The serialized gren object to find

    Returns:
        Tuple of (namespace, gren_hash, result_status) if found, None otherwise
    """
    full_class_name = gren_obj.get("__class__")
    if not isinstance(full_class_name, str):
        return None

    # Convert class name to potential namespace path
    # e.g., "my_project.pipelines.TrainModel" -> "my_project/pipelines/TrainModel"
    namespace_path = Path(*full_class_name.split("."))

    for root in _iter_roots():
        class_dir = root / namespace_path
        if not class_dir.exists():
            continue

        # Search through experiments of this class
        for experiment_dir in _find_experiment_dirs(class_dir):
            metadata = MetadataManager.read_metadata_raw(experiment_dir)
            if not metadata:
                continue

            migration = MigrationManager.read_migration(experiment_dir)
            if migration is not None and migration.kind == "alias":
                continue

            stored_gren_obj = metadata.get("gren_obj")
            if stored_gren_obj == gren_obj:
                namespace, gren_hash = _parse_namespace_from_path(experiment_dir, root)
                state = StateManager.read_state(experiment_dir)
                return namespace, gren_hash, state.result.status

    return None


def get_experiment_relationships(
    namespace: str,
    gren_hash: str,
    *,
    view: str = "resolved",
) -> ExperimentRelationships | None:
    """
    Get parent and child relationships for an experiment.

    Args:
        namespace: Dot-separated namespace (e.g., "my_project.pipelines.TrainModel")
        gren_hash: Hash identifying the specific experiment

    Returns:
        ExperimentRelationships or None if experiment not found
    """
    # First get the experiment's metadata
    namespace_path = Path(*namespace.split("."))

    target_metadata: JsonDict | None = None

    for root in _iter_roots():
        experiment_dir = root / namespace_path / gren_hash
        state_path = StateManager.get_state_path(experiment_dir)

        if state_path.is_file():
            migration = MigrationManager.read_migration(experiment_dir)
            if (
                view == "original"
                and migration is not None
                and migration.kind == "alias"
            ):
                experiment_dir = MigrationManager.resolve_dir(migration, target="from")
            target_metadata = MetadataManager.read_metadata_raw(experiment_dir)
            break

    if not target_metadata:
        return None

    target_gren_obj_raw = target_metadata.get("gren_obj")
    if not isinstance(target_gren_obj_raw, dict):
        return None
    target_gren_obj = cast(dict[str, object], target_gren_obj_raw)

    # Extract parents from this experiment's gren_obj
    parents: list[ParentExperiment] = []
    dependencies = _extract_dependencies_from_gren_obj(target_gren_obj)

    for field_name, dep_class in dependencies:
        parent_obj = target_gren_obj.get(field_name)
        if not isinstance(parent_obj, dict):
            continue

        parent_obj_dict = cast(dict[str, object], parent_obj)
        short_class_name = dep_class.split(".")[-1]

        # Try to find the actual experiment
        found = _find_experiment_by_gren_obj(parent_obj_dict)

        # Extract config (everything except __class__)
        config = {k: v for k, v in parent_obj_dict.items() if k != "__class__"}

        if found:
            parent_ns, parent_hash, parent_status = found
            parents.append(
                ParentExperiment(
                    field_name=field_name,
                    class_name=short_class_name,
                    full_class_name=dep_class,
                    namespace=parent_ns,
                    gren_hash=parent_hash,
                    result_status=parent_status,
                    config=config,
                )
            )
        else:
            parents.append(
                ParentExperiment(
                    field_name=field_name,
                    class_name=short_class_name,
                    full_class_name=dep_class,
                    namespace=None,
                    gren_hash=None,
                    result_status=None,
                    config=config,
                )
            )

    # Find children by scanning all experiments
    children: list[ChildExperiment] = []

    for root in _iter_roots():
        for experiment_dir in _find_experiment_dirs(root):
            migration = MigrationManager.read_migration(experiment_dir)
            if migration is not None and migration.kind == "alias":
                continue

            child_namespace, child_hash = _parse_namespace_from_path(
                experiment_dir, root
            )

            # Skip self
            if child_namespace == namespace and child_hash == gren_hash:
                continue

            child_metadata = MetadataManager.read_metadata_raw(experiment_dir)
            if not child_metadata:
                continue

            child_gren_obj = child_metadata.get("gren_obj")
            if not isinstance(child_gren_obj, dict):
                continue

            child_obj_dict = cast(dict[str, object], child_gren_obj)

            # Check if this experiment depends on our target
            for field_name, value in child_obj_dict.items():
                if field_name == "__class__":
                    continue

                if isinstance(value, dict) and value == target_gren_obj:
                    # This experiment depends on our target
                    child_class = child_obj_dict.get("__class__")
                    if not isinstance(child_class, str):
                        continue

                    state = StateManager.read_state(experiment_dir)
                    children.append(
                        ChildExperiment(
                            field_name=field_name,
                            class_name=child_class.split(".")[-1],
                            full_class_name=child_class,
                            namespace=child_namespace,
                            gren_hash=child_hash,
                            result_status=state.result.status,
                        )
                    )

    return ExperimentRelationships(parents=parents, children=children)
