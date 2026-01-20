"""Filesystem scanner for discovering and parsing Furu experiment state."""

import datetime as _dt
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from ..config import FURU_CONFIG
from ..storage import MetadataManager, MigrationManager, MigrationRecord, StateAttempt
from ..storage.state import StateManager, _FuruState
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
    """Iterate over all existing Furu storage roots."""
    for version_controlled in (False, True):
        root = FURU_CONFIG.get_root(version_controlled)
        if root.exists():
            yield root


def _parse_namespace_from_path(experiment_dir: Path, root: Path) -> tuple[str, str]:
    """
    Parse namespace and furu_hash from experiment directory path.

    Example: /data/my_project/pipelines/TrainModel/abc123 -> ("my_project.pipelines.TrainModel", "abc123")
    """
    relative = experiment_dir.relative_to(root)
    parts = relative.parts
    if len(parts) < 2:  # TODO: Maybe this should throw?
        return str(relative), ""
    furu_hash = parts[-1]
    namespace = ".".join(parts[:-1])
    return namespace, furu_hash


def _alias_key(migration: MigrationRecord) -> tuple[str, str, str]:
    return (migration.from_namespace, migration.from_hash, migration.from_root)


def _collect_aliases() -> dict[tuple[str, str, str], list[MigrationRecord]]:
    aliases: dict[tuple[str, str, str], list[MigrationRecord]] = defaultdict(list)
    for root in _iter_roots():
        for experiment_dir in _find_experiment_dirs(root):
            migration = MigrationManager.read_migration(experiment_dir)
            if migration is None or migration.kind != "alias":
                continue
            if migration.overwritten_at is not None:
                continue
            aliases[_alias_key(migration)].append(migration)
    return aliases


def _alias_reference(
    aliases: dict[tuple[str, str, str], list[MigrationRecord]],
) -> dict[str, dict[str, list[str]]]:
    ref: dict[str, dict[str, list[str]]] = {}
    for key, records in aliases.items():
        from_namespace, from_hash, _from_root = key
        namespace_map = ref.setdefault(from_namespace, {})
        namespace_map[from_hash] = [record.to_hash for record in records]
    return ref


def _get_class_name(namespace: str) -> str:
    """Extract class name from namespace (last component)."""
    parts = namespace.split(".")
    return parts[-1] if parts else namespace


def _migration_kind(migration: MigrationRecord | None) -> str | None:
    if migration is None:
        return None
    if migration.kind == "migrated":
        return None
    return migration.kind


def _override_summary_attempts(
    summary: ExperimentSummary, state: _FuruState
) -> ExperimentSummary:
    attempt = state.attempt
    return summary.model_copy(
        update={
            "attempt_status": attempt.status if attempt else None,
            "attempt_number": attempt.number if attempt else None,
            "backend": attempt.backend if attempt else None,
            "hostname": attempt.owner.hostname if attempt else None,
            "user": attempt.owner.user if attempt else None,
            "started_at": attempt.started_at if attempt else None,
        }
    )


def _state_to_summary(
    state: _FuruState,
    namespace: str,
    furu_hash: str,
    migration: MigrationRecord | None = None,
    original_status: str | None = None,
    original_namespace: str | None = None,
    original_hash: str | None = None,
) -> ExperimentSummary:
    """Convert a Furu state to an experiment summary."""
    attempt = state.attempt
    return ExperimentSummary(
        namespace=namespace,
        furu_hash=furu_hash,
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
        migration_kind=_migration_kind(migration) if migration else None,
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


def _find_experiment_dirs(root: Path) -> list[Path]:
    """Find all directories containing .furu/state.json files."""
    experiments = []

    # Walk the directory tree looking for .furu directories
    for furu_dir in root.rglob(StateManager.INTERNAL_DIR):
        if furu_dir.is_dir():
            state_file = furu_dir / StateManager.STATE_FILE
            if state_file.is_file():
                experiments.append(furu_dir.parent)

    return experiments


def _parse_datetime(value: str | None) -> _dt.datetime | None:
    """Parse ISO datetime string to datetime object."""
    if not value:
        return None
    dt = _dt.datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _read_metadata_with_defaults(
    directory: Path, migration: MigrationRecord | None
) -> JsonDict | None:
    metadata = MetadataManager.read_metadata_raw(directory)
    if not metadata or migration is None:
        return metadata
    if migration.kind != "alias" or migration.overwritten_at is not None:
        return metadata
    if not migration.default_values:
        return metadata

    furu_obj = metadata.get("furu_obj")
    if not isinstance(furu_obj, dict):
        return metadata

    defaults = migration.default_values
    updates: dict[str, str | int | float | bool] = {}
    for field, value in defaults.items():
        if field not in furu_obj:
            updates[field] = value

    if not updates:
        return metadata

    updated_obj = dict(furu_obj)
    updated_obj.update(updates)
    updated_metadata = dict(metadata)
    updated_metadata["furu_obj"] = updated_obj
    return updated_metadata


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
    migration_kind: str | None = None,
    migration_policy: str | None = None,
    view: str = "resolved",
) -> list[ExperimentSummary]:
    """
    Scan the filesystem for Furu experiments.

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
    seen_original: set[tuple[str, str, str]] = set()

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
            namespace, furu_hash = _parse_namespace_from_path(experiment_dir, root)
            migration = MigrationManager.read_migration(experiment_dir)
            original_status: str | None = None
            original_state: _FuruState | None = None
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
                original_key = (
                    migration.from_namespace,
                    migration.from_hash,
                    migration.from_root,
                )
                if view == "original":
                    if original_key in seen_original:
                        continue
                    seen_original.add(original_key)
                    state = original_state
                    namespace = migration.from_namespace
                    furu_hash = migration.from_hash
                    metadata_dir = original_dir
                elif alias_active:
                    metadata_dir = original_dir
            elif view == "original":
                original_key = (
                    namespace,
                    furu_hash,
                    MigrationManager.root_kind_for_dir(experiment_dir),
                )
                if original_key in seen_original:
                    continue
                seen_original.add(original_key)

            summary = _state_to_summary(
                state,
                namespace,
                furu_hash,
                migration=migration,
                original_status=original_status,
                original_namespace=migration.from_namespace if migration else None,
                original_hash=migration.from_hash if migration else None,
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
            if migration_kind and summary.migration_kind != migration_kind:
                continue
            if migration_policy and summary.migration_policy != migration_policy:
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
                defaults_migration = migration if view == "resolved" else None
                metadata = _read_metadata_with_defaults(
                    metadata_dir,
                    defaults_migration,
                )
                if metadata:
                    furu_obj = metadata.get("furu_obj")
                    if isinstance(furu_obj, dict):
                        actual_value = _get_nested_value(furu_obj, config_field)
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
    furu_hash: str,
    *,
    view: str = "resolved",
) -> ExperimentDetail | None:
    """
    Get detailed information about a specific experiment.

    Args:
        namespace: Dot-separated namespace (e.g., "my_project.pipelines.TrainModel")
        furu_hash: Hash identifying the specific experiment
        view: "resolved" uses alias metadata; "original" uses original metadata/state.

    Returns:
        Experiment detail or None if not found
    """
    # Convert namespace to path
    namespace_path = Path(*namespace.split("."))
    alias_reference = _alias_reference(_collect_aliases())

    for root in _iter_roots():
        experiment_dir = root / namespace_path / furu_hash
        state_path = StateManager.get_state_path(experiment_dir)

        if not state_path.is_file():
            continue

        state = StateManager.read_state(experiment_dir)
        migration = MigrationManager.read_migration(experiment_dir)
        metadata = _read_metadata_with_defaults(
            experiment_dir,
            migration if view == "resolved" else None,
        )
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
                furu_hash = original_hash
            else:
                metadata = _read_metadata_with_defaults(original_dir, migration)
        elif migration is not None and migration.kind in {
            "moved",
            "copied",
            "migrated",
        }:
            if view == "original":
                original_dir = MigrationManager.resolve_dir(migration, target="from")
                state = StateManager.read_state(original_dir)
                metadata = MetadataManager.read_metadata_raw(original_dir)
                experiment_dir = original_dir
                namespace = migration.from_namespace
                furu_hash = migration.from_hash
                original_namespace = migration.from_namespace
                original_hash = migration.from_hash

        attempt = state.attempt
        if view == "original" and migration is not None and migration.kind == "alias":
            alias_source_namespace = migration.from_namespace
            alias_source_hash = migration.from_hash
        else:
            alias_source_namespace = namespace
            alias_source_hash = furu_hash

        alias_keys = alias_reference.get(alias_source_namespace, {}).get(
            alias_source_hash,
            [],
        )
        alias_namespaces = (
            [alias_source_namespace] * len(alias_keys) if alias_keys else None
        )
        alias_hashes = alias_keys if alias_keys else None
        return ExperimentDetail(
            namespace=namespace,
            furu_hash=furu_hash,
            class_name=_get_class_name(namespace),
            result_status=state.result.status,
            attempt_status=attempt.status if attempt else None,
            attempt_number=attempt.number if attempt else None,
            updated_at=state.updated_at,
            started_at=attempt.started_at if attempt else None,
            backend=attempt.backend if attempt else None,
            hostname=attempt.owner.hostname if attempt else None,
            user=attempt.owner.user if attempt else None,
            directory=str(experiment_dir),
            state=state.model_dump(mode="json"),
            metadata=metadata,
            attempt=StateAttempt.from_internal(attempt) if attempt else None,
            migration_kind=_migration_kind(migration) if migration else None,
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
            alias_namespaces=alias_namespaces,
            alias_hashes=alias_hashes,
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


def _extract_dependencies_from_furu_obj(
    furu_obj: dict[str, object],
) -> list[tuple[str, str]]:
    """
    Extract dependency class names from a serialized furu object.

    Looks for nested objects with __class__ markers, which indicate Furu dependencies.

    Args:
        furu_obj: The serialized furu object (from metadata.furu_obj)

    Returns:
        List of (field_name, dependency_class_name) tuples
    """
    dependencies: list[tuple[str, str]] = []

    for key, value in furu_obj.items():
        if key == "__class__":
            continue
        if isinstance(value, dict):
            nested_obj = cast(dict[str, object], value)
            dep_class_value = nested_obj.get("__class__")
            if dep_class_value is not None:
                # This is a nested Furu object (dependency)
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
            namespace, furu_hash = _parse_namespace_from_path(experiment_dir, root)
            metadata = MetadataManager.read_metadata_raw(experiment_dir)

            if not metadata:
                continue

            migration = MigrationManager.read_migration(experiment_dir)
            if migration is not None and migration.kind == "alias":
                continue

            furu_obj = metadata.get("furu_obj")
            if not isinstance(furu_obj, dict):
                continue

            full_class_name = furu_obj.get("__class__")
            if not isinstance(full_class_name, str):
                continue

            # Extract short class name
            short_class_name = full_class_name.split(".")[-1]
            class_info[full_class_name] = short_class_name

            # Get attempt status
            attempt_status = state.attempt.status if state.attempt else None

            # Store experiment info
            experiments_by_class[full_class_name].append(
                (namespace, furu_hash, state.result.status, attempt_status)
            )

            # Extract dependencies and create edges
            dependencies = _extract_dependencies_from_furu_obj(furu_obj)
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
                    furu_hash=h,
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


def _find_experiment_by_furu_obj(
    furu_obj: dict[str, object],
) -> tuple[str, str, str] | None:
    """
    Find an experiment that matches the given furu_obj serialization.

    Args:
        furu_obj: The serialized furu object to find

    Returns:
        Tuple of (namespace, furu_hash, result_status) if found, None otherwise
    """
    full_class_name = furu_obj.get("__class__")
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

            stored_furu_obj = metadata.get("furu_obj")
            if stored_furu_obj == furu_obj:
                namespace, furu_hash = _parse_namespace_from_path(experiment_dir, root)
                state = StateManager.read_state(experiment_dir)
                return namespace, furu_hash, state.result.status

    return None


def get_experiment_relationships(
    namespace: str,
    furu_hash: str,
    *,
    view: str = "resolved",
) -> ExperimentRelationships | None:
    """
    Get parent and child relationships for an experiment.

    Args:
        namespace: Dot-separated namespace (e.g., "my_project.pipelines.TrainModel")
        furu_hash: Hash identifying the specific experiment

    Returns:
        ExperimentRelationships or None if experiment not found
    """
    # First get the experiment's metadata
    namespace_path = Path(*namespace.split("."))

    target_metadata: JsonDict | None = None

    for root in _iter_roots():
        experiment_dir = root / namespace_path / furu_hash
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
            else:
                target_metadata = _read_metadata_with_defaults(
                    experiment_dir,
                    migration if view == "resolved" else None,
                )
            break

    if not target_metadata:
        return None

    target_furu_obj_raw = target_metadata.get("furu_obj")
    if not isinstance(target_furu_obj_raw, dict):
        return None
    target_furu_obj = cast(dict[str, object], target_furu_obj_raw)

    # Extract parents from this experiment's furu_obj
    parents: list[ParentExperiment] = []
    dependencies = _extract_dependencies_from_furu_obj(target_furu_obj)

    for field_name, dep_class in dependencies:
        parent_obj = target_furu_obj.get(field_name)
        if not isinstance(parent_obj, dict):
            continue

        parent_obj_dict = cast(dict[str, object], parent_obj)
        short_class_name = dep_class.split(".")[-1]

        # Try to find the actual experiment
        found = _find_experiment_by_furu_obj(parent_obj_dict)

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
                    furu_hash=parent_hash,
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
                    furu_hash=None,
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
            if child_namespace == namespace and child_hash == furu_hash:
                continue

            child_metadata = MetadataManager.read_metadata_raw(experiment_dir)
            if not child_metadata:
                continue

            child_furu_obj = child_metadata.get("furu_obj")
            if not isinstance(child_furu_obj, dict):
                continue

            child_obj_dict = cast(dict[str, object], child_furu_obj)

            # Check if this experiment depends on our target
            for field_name, value in child_obj_dict.items():
                if field_name == "__class__":
                    continue

                if isinstance(value, dict) and value == target_furu_obj:
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
                            furu_hash=child_hash,
                            result_status=state.result.status,
                        )
                    )

    return ExperimentRelationships(parents=parents, children=children)
