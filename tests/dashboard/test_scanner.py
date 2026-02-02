"""Tests for the Furu experiment scanner."""

from __future__ import annotations

import json
from pathlib import Path

from furu.dashboard.scanner import (
    get_experiment_dag,
    get_experiment_detail,
    get_stats,
    scan_experiments,
)
from furu.serialization import FuruSerializer
from furu.storage import MigrationManager

from .conftest import create_experiment_from_furu
from .pipelines import PrepareDataset, TrainModel


class VersionedDataset(PrepareDataset, version_controlled=True):
    """PrepareDataset variant stored in version-controlled root."""


def test_scan_experiments_empty(temp_furu_root: Path) -> None:
    """Test scanning when no experiments exist."""
    experiments = scan_experiments()
    assert experiments == []


def test_scan_experiments_finds_all(populated_furu_root: Path) -> None:
    """Test that scanner finds all experiments."""
    experiments = scan_experiments()
    # 7 experiments (current schema): dataset1, train1, train2, eval1, loader, alias, alias2
    assert len(experiments) == 7


def test_scan_experiments_schema_filter(populated_furu_root: Path) -> None:
    current = scan_experiments()
    assert len(current) == 7

    any_schema = scan_experiments(schema="any")
    assert len(any_schema) == 8

    stale = scan_experiments(schema="stale")
    assert len(stale) == 1
    assert stale[0].is_stale is True


def test_scan_experiments_filter_result_status(populated_furu_root: Path) -> None:
    """Test filtering by result status."""
    experiments = scan_experiments(result_status="success")
    # 3 successful: dataset1, train1, loader
    assert len(experiments) == 3
    for exp in experiments:
        assert exp.result_status == "success"

    migrated = scan_experiments(result_status="migrated")
    assert len(migrated) == 2
    assert {exp.migration_kind for exp in migrated} == {"alias"}

    alias_policy = scan_experiments(
        result_status="migrated",
        migration_policy="alias",
    )
    assert len(alias_policy) == 2
    assert all(exp.migration_policy == "alias" for exp in alias_policy)


def test_scan_experiments_filter_attempt_status(populated_furu_root: Path) -> None:
    """Test filtering by attempt status."""
    experiments = scan_experiments(attempt_status="failed")
    assert len(experiments) == 1
    assert experiments[0].attempt_status == "failed"

    resolved = scan_experiments(attempt_status="failed", view="resolved")
    assert len(resolved) == 1


def test_scan_experiments_filter_namespace(populated_furu_root: Path) -> None:
    """Test filtering by namespace prefix."""
    experiments = scan_experiments(namespace_prefix="dashboard.pipelines")
    # All 7 current-schema experiments are in dashboard.pipelines
    assert len(experiments) == 7
    for exp in experiments:
        assert exp.namespace.startswith("dashboard.pipelines")

    original = scan_experiments(namespace_prefix="dashboard.pipelines", view="original")
    assert len(original) == 5


def test_scan_experiments_sorted_by_updated_at(temp_furu_root: Path) -> None:
    """Test that experiments are sorted by updated_at (newest first)."""
    # Create experiments with different timestamps
    older_dataset = PrepareDataset(name="older", version="v1")
    older_dir = create_experiment_from_furu(
        older_dataset, result_status="success", attempt_status="success"
    )

    # Modify the state file to have an older timestamp
    older_state = older_dir / ".furu" / "state.json"
    state_data = json.loads(older_state.read_text())
    state_data["updated_at"] = "2024-01-01T00:00:00+00:00"
    older_state.write_text(json.dumps(state_data))

    newer_dataset = PrepareDataset(name="newer", version="v1")
    newer_dir = create_experiment_from_furu(
        newer_dataset, result_status="success", attempt_status="success"
    )

    newer_state = newer_dir / ".furu" / "state.json"
    state_data = json.loads(newer_state.read_text())
    state_data["updated_at"] = "2025-06-01T00:00:00+00:00"
    newer_state.write_text(json.dumps(state_data))

    experiments = scan_experiments()
    assert len(experiments) == 2
    # Newer should come first
    assert experiments[0].furu_hash == FuruSerializer.compute_hash(newer_dataset)
    assert experiments[1].furu_hash == FuruSerializer.compute_hash(older_dataset)


def test_get_experiment_detail_found(populated_furu_root: Path) -> None:
    """Test getting experiment detail."""
    dataset1 = PrepareDataset(name="mnist", version="v1")
    furu_hash = FuruSerializer.compute_hash(dataset1)

    detail = get_experiment_detail("dashboard.pipelines.PrepareDataset", furu_hash)
    assert detail is not None
    assert detail.namespace == "dashboard.pipelines.PrepareDataset"
    assert detail.furu_hash == furu_hash
    assert detail.result_status == "success"
    assert detail.metadata is not None
    assert "state" in detail.model_dump()

    alias = PrepareDataset(name="mnist", version="v2")
    alias_hash = FuruSerializer.compute_hash(alias)
    alias_detail = get_experiment_detail(
        "dashboard.pipelines.PrepareDataset", alias_hash, view="resolved"
    )
    assert alias_detail is not None
    assert alias_detail.migration_kind == "alias"
    assert alias_detail.original_result_status == "success"
    assert alias_detail.is_alias is True
    assert alias_detail.aliases is None

    assert detail.aliases is not None
    assert len(detail.aliases) == 2

    alias_dir = alias._base_furu_dir()
    alias_record = MigrationManager.read_migration(alias_dir)
    assert alias_record is not None
    MigrationManager.write_migration(
        alias_record.model_copy(update={"overwritten_at": "2025-01-06T00:00:00+00:00"}),
        alias_dir,
    )
    alias_filtered = get_experiment_detail(
        "dashboard.pipelines.PrepareDataset", furu_hash, view="resolved"
    )
    assert alias_filtered is not None
    assert alias_filtered.aliases is not None
    assert len(alias_filtered.aliases) == 2
    assert any(alias.overwritten_at is not None for alias in alias_filtered.aliases)

    MigrationManager.write_migration(alias_record, alias_dir)

    alias_original = get_experiment_detail(
        "dashboard.pipelines.PrepareDataset", alias_hash, view="original"
    )
    assert alias_original is not None
    assert alias_original.furu_hash == furu_hash


def test_get_experiment_detail_not_found(populated_furu_root: Path) -> None:
    """Test getting detail for a non-existent experiment."""
    detail = get_experiment_detail("nonexistent.Namespace", "fakehash")
    assert detail is None


def test_get_experiment_detail_includes_attempt(populated_furu_root: Path) -> None:
    """Test that detail includes attempt information."""
    dataset1 = PrepareDataset(name="mnist", version="v1")
    train2 = TrainModel(lr=0.0001, steps=2000, dataset=dataset1)
    furu_hash = FuruSerializer.compute_hash(train2)

    detail = get_experiment_detail("dashboard.pipelines.TrainModel", furu_hash)
    assert detail is not None
    assert detail.attempt is not None
    assert detail.attempt.status == "running"
    assert detail.attempt.owner.host == "gpu-02"  # From populated fixture


def test_get_stats_empty(temp_furu_root: Path) -> None:
    """Test stats with no experiments."""
    stats = get_stats()
    assert stats.total == 0
    assert stats.running_count == 0
    assert stats.success_count == 0


def test_get_stats_counts(populated_furu_root: Path) -> None:
    """Test that stats correctly count experiments."""
    stats = get_stats()
    # 8 total: dataset1(success), train1(success), train2(running),
    #          eval1(failed), loader(success), dataset2(absent stale),
    #          alias(migrated), alias2(migrated)
    assert stats.total == 8
    assert stats.success_count == 3
    assert stats.failed_count == 1
    assert stats.running_count == 1

    # Check by_result_status
    result_map = {s.status: s.count for s in stats.by_result_status}
    assert result_map["success"] == 3
    assert result_map["failed"] == 1
    assert result_map["incomplete"] == 1
    assert result_map.get("absent", 0) == 1
    assert result_map["migrated"] == 2


def test_scan_experiments_version_controlled(temp_furu_root: Path) -> None:
    """Test that scanner finds experiments in both roots."""
    # Create an unversioned experiment
    unversioned = PrepareDataset(name="unversioned", version="v1")
    create_experiment_from_furu(unversioned, result_status="success")

    versioned = VersionedDataset(name="versioned", version="v1")
    create_experiment_from_furu(versioned, result_status="success")

    experiments = scan_experiments()
    assert len(experiments) >= 2
    namespaces = {exp.namespace for exp in experiments}
    assert "dashboard.pipelines.PrepareDataset" in namespaces
    expected_versioned = (
        f"{VersionedDataset.__module__}.{VersionedDataset.__qualname__}"
    )
    assert expected_versioned in namespaces


def test_experiment_summary_class_name(temp_furu_root: Path) -> None:
    """Test that class_name is correctly extracted from namespace."""
    dataset = PrepareDataset(name="test", version="v1")
    create_experiment_from_furu(dataset, result_status="success")

    experiments = scan_experiments()
    assert len(experiments) == 1
    assert experiments[0].class_name == "PrepareDataset"


def test_scan_experiments_filter_by_class(populated_furu_root: Path) -> None:
    """Test filtering experiments by specific class."""
    experiments = scan_experiments(namespace_prefix="dashboard.pipelines.TrainModel")
    # 2 TrainModel experiments: train1 and train2
    assert len(experiments) == 2
    for exp in experiments:
        assert exp.class_name == "TrainModel"


# =============================================================================
# Tests for new filtering features: backend, hostname, user, date range, config
# =============================================================================


def test_scan_experiments_filter_by_backend(populated_furu_root: Path) -> None:
    """Test filtering experiments by backend."""
    # Fixture has: dataset1(local), train1(local), train2(submitit),
    #              eval1(local), loader(submitit), dataset2(no attempt), alias(resolves)

    # Filter by local backend
    local_results = scan_experiments(backend="local")
    assert len(local_results) == 5  # dataset1, train1, eval1, aliases
    for exp in local_results:
        assert exp.backend == "local"

    # Filter by submitit backend
    submitit_results = scan_experiments(backend="submitit")
    assert len(submitit_results) == 2  # train2, loader
    for exp in submitit_results:
        assert exp.backend == "submitit"


def test_scan_experiments_filter_by_backend_no_match(temp_furu_root: Path) -> None:
    """Test filtering by backend returns empty when no experiments match."""
    exp = PrepareDataset(name="test", version="v1")
    create_experiment_from_furu(
        exp,
        result_status="success",
        attempt_status="success",
        backend="local",
    )

    results = scan_experiments(backend="submitit")
    assert len(results) == 0


def test_scan_experiments_filter_by_hostname(populated_furu_root: Path) -> None:
    """Test filtering experiments by hostname."""
    # Fixture has: dataset1(gpu-01), train1(gpu-01), train2(gpu-02),
    #              eval1(gpu-02), loader(gpu-01), dataset2(no attempt), alias(resolves)

    # Filter by gpu-01
    gpu01_results = scan_experiments(hostname="gpu-01")
    assert len(gpu01_results) == 5  # dataset1, train1, loader, aliases
    for exp in gpu01_results:
        assert exp.hostname == "gpu-01"

    # Filter by gpu-02
    gpu02_results = scan_experiments(hostname="gpu-02")
    assert len(gpu02_results) == 2  # train2, eval1
    for exp in gpu02_results:
        assert exp.hostname == "gpu-02"


def test_scan_experiments_filter_by_hostname_no_match(temp_furu_root: Path) -> None:
    """Test filtering by hostname returns empty when no experiments match."""
    exp = PrepareDataset(name="test", version="v1")
    create_experiment_from_furu(
        exp,
        result_status="success",
        attempt_status="success",
        hostname="existing-host",
    )

    results = scan_experiments(hostname="nonexistent-host")
    assert len(results) == 0


def test_scan_experiments_filter_by_user(populated_furu_root: Path) -> None:
    """Test filtering experiments by user."""
    # Fixture has: dataset1(alice), train1(alice), train2(bob),
    #              eval1(alice), loader(bob), dataset2(no attempt), alias(resolves)

    # Filter by alice
    alice_results = scan_experiments(user="alice")
    assert len(alice_results) == 5  # dataset1, train1, eval1, aliases
    for exp in alice_results:
        assert exp.user == "alice"

    # Filter by bob
    bob_results = scan_experiments(user="bob")
    assert len(bob_results) == 2  # train2, loader
    for exp in bob_results:
        assert exp.user == "bob"


def test_scan_experiments_filter_by_user_no_match(temp_furu_root: Path) -> None:
    """Test filtering by user returns empty when no experiments match."""
    exp = PrepareDataset(name="test", version="v1")
    create_experiment_from_furu(
        exp,
        result_status="success",
        attempt_status="success",
        user="existinguser",
    )

    results = scan_experiments(user="nonexistentuser")
    assert len(results) == 0


def test_scan_experiments_filter_by_started_after(populated_furu_root: Path) -> None:
    """Test filtering experiments started after a specific time."""
    # Fixture has: loader(2024-06-01), dataset1(2025-01-01), train1(2025-01-02),
    #              train2(2025-01-03), eval1(2025-01-04), dataset2(no attempt)

    # Filter by started_after 2025-01-01 should exclude loader (2024) and dataset2 (no attempt)
    results = scan_experiments(started_after="2025-01-01T00:00:00+00:00")
    assert len(results) == 6  # dataset1, train1, train2, eval1, aliases
    for exp in results:
        assert exp.started_at is not None
        assert exp.started_at >= "2025-01-01T00:00:00+00:00"


def test_scan_experiments_filter_by_started_before(populated_furu_root: Path) -> None:
    """Test filtering experiments started before a specific time."""
    # Fixture has: loader(2024-06-01), dataset1(2025-01-01), train1(2025-01-02),
    #              train2(2025-01-03), eval1(2025-01-04), dataset2(no attempt)

    # Filter by started_before 2025-01-01 should only get loader
    results = scan_experiments(started_before="2025-01-01T00:00:00+00:00")
    assert len(results) == 1  # loader only
    assert results[0].started_at == "2024-06-01T10:00:00+00:00"


def test_scan_experiments_filter_by_started_range(populated_furu_root: Path) -> None:
    """Test filtering experiments within a date range."""
    # Fixture has: loader(2024-06-01), dataset1(2025-01-01), train1(2025-01-02),
    #              train2(2025-01-03), eval1(2025-01-04), dataset2(no attempt)

    # Filter to get only train1 and train2 (2025-01-02 to 2025-01-03)
    results = scan_experiments(
        started_after="2025-01-02T00:00:00+00:00",
        started_before="2025-01-04T00:00:00+00:00",
    )
    assert len(results) == 2  # train1, train2
    started_dates = {exp.started_at for exp in results}
    assert "2025-01-02T10:00:00+00:00" in started_dates
    assert "2025-01-03T10:00:00+00:00" in started_dates


def test_scan_experiments_filter_by_updated_after(populated_furu_root: Path) -> None:
    """Test filtering experiments updated after a specific time."""
    # Fixture has: loader(updated 2024-06-01), dataset1(2025-01-01), train1(2025-01-02),
    #              train2(2025-01-03), eval1(2025-01-04), dataset2(no attempt with default date)

    # Filter by updated_after 2025-01-02 should get train1, train2, eval1
    results = scan_experiments(updated_after="2025-01-02T00:00:00+00:00")
    assert len(results) == 3  # train1, train2, eval1
    for exp in results:
        assert exp.updated_at is not None
        assert exp.updated_at >= "2025-01-02T00:00:00+00:00"


def test_scan_experiments_filter_by_updated_before(populated_furu_root: Path) -> None:
    """Test filtering experiments updated before a specific time."""
    # Fixture has: loader(updated 2024-06-01), dataset1(2025-01-01), train1(2025-01-02),
    #              train2(2025-01-03), eval1(2025-01-04), dataset2(default)

    # Filter by updated_before 2025-01-01 should only get loader (2024-06-01)
    results = scan_experiments(updated_before="2025-01-01T00:00:00+00:00")
    assert len(results) == 1
    assert results[0].updated_at == "2024-06-01T11:00:00+00:00"


def test_scan_experiments_filter_by_config_field(temp_furu_root: Path) -> None:
    """Test filtering experiments by a config field value."""
    # Create experiments with different config values
    mnist_exp = PrepareDataset(name="mnist", version="v1")
    create_experiment_from_furu(
        mnist_exp,
        result_status="success",
        attempt_status="success",
    )

    cifar_exp = PrepareDataset(name="cifar", version="v1")
    create_experiment_from_furu(
        cifar_exp,
        result_status="success",
        attempt_status="success",
    )

    # Filter by config field (name=mnist)
    results = scan_experiments(config_filter="name=mnist")
    assert len(results) == 1

    # Filter by config field (name=cifar)
    results = scan_experiments(config_filter="name=cifar")
    assert len(results) == 1


def test_scan_experiments_filter_by_config_field_no_match(
    temp_furu_root: Path,
) -> None:
    """Test filtering by config field returns empty when no experiments match."""
    exp = PrepareDataset(name="mnist", version="v1")
    create_experiment_from_furu(
        exp,
        result_status="success",
        attempt_status="success",
    )

    results = scan_experiments(config_filter="name=nonexistent")
    assert len(results) == 0


def test_scan_experiments_filter_by_nested_config_field(temp_furu_root: Path) -> None:
    """Test filtering experiments by a nested config field value."""
    # Create TrainModel experiments with different learning rates
    dataset = PrepareDataset(name="mnist", version="v1")
    create_experiment_from_furu(
        dataset,
        result_status="success",
        attempt_status="success",
    )

    train1 = TrainModel(lr=0.001, steps=1000, dataset=dataset)
    create_experiment_from_furu(
        train1,
        result_status="success",
        attempt_status="success",
    )

    train2 = TrainModel(lr=0.01, steps=500, dataset=dataset)
    create_experiment_from_furu(
        train2,
        result_status="success",
        attempt_status="success",
    )

    # Filter by nested config field (lr=0.001) - note values are stringified
    results = scan_experiments(config_filter="lr=0.001")
    assert len(results) == 1

    # Filter by steps
    results = scan_experiments(config_filter="steps=500")
    assert len(results) == 1


def test_scan_experiments_combined_filters(populated_furu_root: Path) -> None:
    """Test combining multiple filters together."""
    # Fixture has:
    # - dataset1: success, local, gpu-01, alice, 2025-01-01
    # - train1: success, local, gpu-01, alice, 2025-01-02
    # - train2: running, submitit, gpu-02, bob, 2025-01-03
    # - eval1: failed, local, gpu-02, alice, 2025-01-04
    # - loader: success, submitit, gpu-01, bob, 2024-06-01
    # - dataset2: absent (stale schema)
    # - alias: migrated (resolves to dataset1)
    # - alias2: migrated (resolves to dataset1)

    # Combine result_status + user: success + alice = dataset1, train1
    results = scan_experiments(result_status="success", user="alice")
    assert len(results) == 2
    for exp in results:
        assert exp.user == "alice"
        assert exp.result_status == "success"

    # Combine backend + hostname: local + gpu-01 = dataset1, train1, aliases
    results = scan_experiments(backend="local", hostname="gpu-01")
    assert len(results) == 4
    for exp in results:
        assert exp.backend == "local"
        assert exp.hostname == "gpu-01"

    # Combine multiple filters: success + submitit + gpu-01 = loader only
    results = scan_experiments(
        result_status="success",
        backend="submitit",
        hostname="gpu-01",
    )
    assert len(results) == 1
    assert results[0].backend == "submitit"
    assert results[0].hostname == "gpu-01"


def test_scan_experiments_combined_filters_no_match(temp_furu_root: Path) -> None:
    """Test combined filters return empty when combination doesn't match."""
    exp = PrepareDataset(name="test", version="v1")
    create_experiment_from_furu(
        exp,
        result_status="success",
        attempt_status="success",
        backend="local",
        hostname="gpu-01",
        user="alice",
    )

    # No experiment matches both conditions
    results = scan_experiments(backend="submitit", user="alice")
    assert len(results) == 0


def test_scan_experiments_filter_experiment_without_attempt(
    temp_furu_root: Path,
) -> None:
    """Test filtering excludes experiments without attempts when filtering by attempt fields."""
    # Create experiment with attempt
    with_attempt = PrepareDataset(name="with_attempt", version="v1")
    create_experiment_from_furu(
        with_attempt,
        result_status="success",
        attempt_status="success",
        backend="local",
        hostname="gpu-01",
    )

    # Create experiment without attempt (absent status)
    without_attempt = PrepareDataset(name="without_attempt", version="v1")
    create_experiment_from_furu(
        without_attempt,
        result_status="absent",
        attempt_status=None,
    )

    # Filtering by backend should exclude experiments without attempts
    results = scan_experiments(backend="local")
    assert len(results) == 1
    assert results[0].backend == "local"

    # Filtering by hostname should exclude experiments without attempts
    results = scan_experiments(hostname="gpu-01")
    assert len(results) == 1

    # All experiments
    all_results = scan_experiments()
    assert len(all_results) == 2


# =============================================================================
# Tests for DAG extraction functionality
# =============================================================================


def test_get_experiment_dag_empty(temp_furu_root: Path) -> None:
    """Test DAG with no experiments."""
    dag = get_experiment_dag()
    assert dag.total_nodes == 0
    assert dag.total_edges == 0
    assert dag.total_experiments == 0
    assert dag.nodes == []
    assert dag.edges == []


def test_get_experiment_dag_single_node(temp_furu_root: Path) -> None:
    """Test DAG with a single experiment (no dependencies)."""
    dataset = PrepareDataset(name="mnist", version="v1")
    create_experiment_from_furu(
        dataset,
        result_status="success",
        attempt_status="success",
    )

    dag = get_experiment_dag()
    assert dag.total_nodes == 1
    assert dag.total_edges == 0
    assert dag.total_experiments == 1

    # Check node properties
    node = dag.nodes[0]
    assert node.class_name == "PrepareDataset"
    assert "PrepareDataset" in node.full_class_name
    assert node.total_count == 1
    assert node.success_count == 1
    assert len(node.experiments) == 1


def test_get_experiment_dag_with_dependency(temp_furu_root: Path) -> None:
    """Test DAG with a simple dependency chain."""
    dataset = PrepareDataset(name="mnist", version="v1")
    create_experiment_from_furu(
        dataset,
        result_status="success",
        attempt_status="success",
    )

    train = TrainModel(lr=0.001, steps=1000, dataset=dataset)
    create_experiment_from_furu(
        train,
        result_status="success",
        attempt_status="success",
    )

    dag = get_experiment_dag()
    assert dag.total_nodes == 2
    assert dag.total_edges == 1
    assert dag.total_experiments == 2

    # Check edge direction (PrepareDataset -> TrainModel)
    edge = dag.edges[0]
    assert "PrepareDataset" in edge.source
    assert "TrainModel" in edge.target
    assert edge.field_name == "dataset"


def test_get_experiment_dag_multiple_experiments_same_class(
    temp_furu_root: Path,
) -> None:
    """Test DAG groups multiple experiments of the same class into one node."""
    dataset1 = PrepareDataset(name="mnist", version="v1")
    create_experiment_from_furu(
        dataset1,
        result_status="success",
        attempt_status="success",
    )

    dataset2 = PrepareDataset(name="cifar", version="v1")
    create_experiment_from_furu(
        dataset2,
        result_status="failed",
        attempt_status="failed",
    )

    dataset3 = PrepareDataset(name="imagenet", version="v1")
    create_experiment_from_furu(
        dataset3,
        result_status="incomplete",
        attempt_status="running",
    )

    dag = get_experiment_dag()
    assert dag.total_nodes == 1  # Single node for all PrepareDataset
    assert dag.total_experiments == 3

    node = dag.nodes[0]
    assert node.class_name == "PrepareDataset"
    assert node.total_count == 3
    assert node.success_count == 1
    assert node.failed_count == 1
    assert node.running_count == 1
    assert len(node.experiments) == 3


def test_get_experiment_dag_populated(populated_furu_root: Path) -> None:
    """Test DAG with the populated fixture data."""
    dag = get_experiment_dag()

    # Fixture has: PrepareDataset (2), TrainModel (2), EvalModel (1), DataLoader (1)
    # Classes: PrepareDataset, TrainModel, EvalModel, DataLoader = 4 nodes
    assert dag.total_nodes == 4
    assert dag.total_experiments == 6

    # Check node counts
    node_by_class = {n.class_name: n for n in dag.nodes}
    assert "PrepareDataset" in node_by_class
    assert "TrainModel" in node_by_class
    assert "EvalModel" in node_by_class
    assert "DataLoader" in node_by_class

    # PrepareDataset has 2 experiments
    assert node_by_class["PrepareDataset"].total_count == 2
    # TrainModel has 2 experiments
    assert node_by_class["TrainModel"].total_count == 2
    # EvalModel has 1 experiment
    assert node_by_class["EvalModel"].total_count == 1
    # DataLoader has 1 experiment
    assert node_by_class["DataLoader"].total_count == 1


def test_get_experiment_dag_edges_populated(populated_furu_root: Path) -> None:
    """Test DAG edges with the populated fixture data."""
    dag = get_experiment_dag()

    # Expected edges:
    # PrepareDataset -> TrainModel (via dataset field)
    # TrainModel -> EvalModel (via model field)
    # So at least 2 edges
    assert dag.total_edges >= 2

    # TrainModel should have an incoming edge from PrepareDataset
    train_edge = next((e for e in dag.edges if "TrainModel" in e.target), None)
    assert train_edge is not None
    assert "PrepareDataset" in train_edge.source
    assert train_edge.field_name == "dataset"

    # EvalModel should have an incoming edge from TrainModel
    eval_edge = next((e for e in dag.edges if "EvalModel" in e.target), None)
    assert eval_edge is not None
    assert "TrainModel" in eval_edge.source
    assert eval_edge.field_name == "model"


def test_get_experiment_dag_with_real_dependencies(
    populated_with_dependencies: Path,
) -> None:
    """Test DAG with experiments that have real dependencies created via get."""
    dag = get_experiment_dag()

    # populated_with_dependencies creates:
    # - 2 PrepareDataset
    # - 1 TrainModel (depends on dataset1)
    # - 1 EvalModel (depends on train)
    # - 1 MultiDependencyPipeline (depends on dataset1 and dataset2)
    assert dag.total_experiments == 5

    # All should be successful since they were created via get
    for node in dag.nodes:
        assert node.success_count == node.total_count

    # Check edges
    # MultiDependencyPipeline has 2 incoming edges (dataset1, dataset2)
    multi_edges = [e for e in dag.edges if "MultiDependencyPipeline" in e.target]
    assert len(multi_edges) == 2
    field_names = {e.field_name for e in multi_edges}
    assert field_names == {"dataset1", "dataset2"}


def test_get_experiment_dag_experiment_details(temp_furu_root: Path) -> None:
    """Test that DAG nodes contain correct experiment details."""
    dataset = PrepareDataset(name="mnist", version="v1")
    furu_hash = FuruSerializer.compute_hash(dataset)

    create_experiment_from_furu(
        dataset,
        result_status="success",
        attempt_status="success",
    )

    dag = get_experiment_dag()
    node = dag.nodes[0]

    # Check experiment details
    exp = node.experiments[0]
    assert exp.furu_hash == furu_hash
    assert exp.namespace == "dashboard.pipelines.PrepareDataset"
    assert exp.result_status == "success"
    assert exp.attempt_status == "success"
