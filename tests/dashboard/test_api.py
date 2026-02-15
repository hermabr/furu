"""Tests for Dashboard API routes.

NOTE: For performance, use module-scoped fixtures (like `populated_furu_root`)
instead of creating experiments in each test with `temp_furu_root`. The shared
fixture creates experiments once and reuses them across all tests in the module.

See `conftest.py:_create_populated_experiments()` for the fixture data setup.
When adding new filter tests, prefer extending the shared fixture data rather
than creating experiments per-test.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from furu.serialization import FuruSerializer
from furu.query import Q

from .pipelines import PrepareDataset, TrainModel


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_list_experiments_empty(client: TestClient, temp_furu_root: Path) -> None:
    """Test listing experiments when none exist."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert data["experiments"] == []
    assert data["total"] == 0


def test_list_experiments(client: TestClient, populated_furu_root: Path) -> None:
    """Test listing all experiments."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    # 7 experiments (current schema): dataset1, train1, train2, eval1, loader, alias, alias2
    assert data["total"] == 7
    assert len(data["experiments"]) == 7

    # Check structure of returned experiments
    exp = data["experiments"][0]
    assert "namespace" in exp
    assert "furu_hash" in exp
    assert "class_name" in exp
    assert "result_status" in exp

    response_original = client.get("/api/experiments?view=original")
    assert response_original.status_code == 200
    original_data = response_original.json()
    assert original_data["total"] == 5


def test_list_experiments_schema_filter(
    client: TestClient, populated_furu_root: Path
) -> None:
    response_current = client.get("/api/experiments")
    assert response_current.status_code == 200
    assert response_current.json()["total"] == 7

    response_stale = client.get("/api/experiments?schema=stale")
    assert response_stale.status_code == 200
    assert response_stale.json()["total"] == 1

    response_any = client.get("/api/experiments?schema=any")
    assert response_any.status_code == 200
    assert response_any.json()["total"] == 8


def test_list_experiments_filter_by_result_status(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by result status."""
    response = client.get("/api/experiments?result_status=success")
    assert response.status_code == 200
    data = response.json()
    # 3 successful: dataset1, train1, loader
    assert data["total"] == 3
    for exp in data["experiments"]:
        assert exp["result_status"] == "success"

    migrated = client.get("/api/experiments?result_status=migrated")
    assert migrated.status_code == 200
    migrated_data = migrated.json()
    assert migrated_data["total"] == 2
    assert {exp["migration_kind"] for exp in migrated_data["experiments"]} == {
        "alias",
    }

    alias_policy = client.get(
        "/api/experiments?result_status=migrated&migration_policy=alias"
    )
    assert alias_policy.status_code == 200
    alias_data = alias_policy.json()
    assert alias_data["total"] == 2
    assert all(exp["migration_policy"] == "alias" for exp in alias_data["experiments"])


def test_list_experiments_filter_by_attempt_status(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by attempt status."""
    response = client.get("/api/experiments?attempt_status=running")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["experiments"][0]["attempt_status"] == "running"


def test_list_experiments_filter_by_namespace(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by namespace prefix."""
    response = client.get("/api/experiments?namespace=dashboard.pipelines")
    assert response.status_code == 200
    data = response.json()
    # All 7 current-schema experiments are in dashboard.pipelines
    assert data["total"] == 7
    for exp in data["experiments"]:
        assert exp["namespace"].startswith("dashboard.pipelines")

    original_view = client.get(
        "/api/experiments?namespace=dashboard.pipelines&view=original"
    )
    assert original_view.status_code == 200
    data = original_view.json()
    assert data["total"] == 5


def test_list_experiments_filter_by_class(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by class name."""
    response = client.get("/api/experiments?namespace=dashboard.pipelines.TrainModel")
    assert response.status_code == 200
    data = response.json()
    # 2 TrainModel experiments: train1 and train2
    assert data["total"] == 2
    for exp in data["experiments"]:
        assert exp["class_name"] == "TrainModel"


def test_list_experiments_pagination(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test pagination of experiments."""
    response = client.get("/api/experiments?limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 7
    assert len(data["experiments"]) == 2

    response = client.get("/api/experiments?limit=2&offset=2")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 7
    assert len(data["experiments"]) == 2


def test_get_experiment_detail(client: TestClient, populated_furu_root: Path) -> None:
    """Test getting detailed experiment information."""
    # Get the hash for a specific experiment
    dataset1 = PrepareDataset(name="mnist", version="v1")
    furu_hash = FuruSerializer.compute_hash(dataset1)

    response = client.get(
        f"/api/experiments/dashboard.pipelines.PrepareDataset/{furu_hash}"
    )
    assert response.status_code == 200
    data = response.json()

    assert data["namespace"] == "dashboard.pipelines.PrepareDataset"
    assert data["furu_hash"] == furu_hash
    assert data["class_name"] == "PrepareDataset"
    assert data["result_status"] == "success"
    assert data["attempt_status"] == "success"
    assert "directory" in data
    assert "state" in data
    assert "metadata" in data

    alias = PrepareDataset(name="mnist", version="v2")
    alias_hash = FuruSerializer.compute_hash(alias)
    alias_response = client.get(
        f"/api/experiments/dashboard.pipelines.PrepareDataset/{alias_hash}?view=resolved"
    )
    assert alias_response.status_code == 200
    alias_data = alias_response.json()
    assert alias_data["migration_kind"] == "alias"
    assert alias_data["original_result_status"] == "success"
    assert alias_data["is_alias"] is True
    assert alias_data["aliases"] is None

    assert data["aliases"] is not None
    assert len(data["aliases"]) == 2


def test_get_experiment_detail_with_attempt(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test that experiment detail includes attempt information."""
    # Get the hash for the running training experiment
    dataset1 = PrepareDataset(name="mnist", version="v1")
    train2 = TrainModel(lr=0.0001, steps=2000, dataset=dataset1)
    furu_hash = FuruSerializer.compute_hash(train2)

    response = client.get(
        f"/api/experiments/dashboard.pipelines.TrainModel/{furu_hash}"
    )
    assert response.status_code == 200
    data = response.json()

    assert data["attempt_status"] == "running"
    assert data["attempt"] is not None
    assert data["attempt"]["status"] == "running"
    assert data["attempt"]["owner"]["host"] == "gpu-02"

    response_original = client.get(
        f"/api/experiments/dashboard.pipelines.TrainModel/{furu_hash}?view=original"
    )
    assert response_original.status_code == 200


def test_get_experiment_not_found(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test getting a non-existent experiment."""
    response = client.get("/api/experiments/nonexistent.Namespace/fake123hash")
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def test_dashboard_stats_empty(client: TestClient, temp_furu_root: Path) -> None:
    """Test stats endpoint with no experiments."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["running_count"] == 0
    assert data["queued_count"] == 0
    assert data["failed_count"] == 0
    assert data["success_count"] == 0


def test_dashboard_stats(client: TestClient, populated_furu_root: Path) -> None:
    """Test aggregate statistics endpoint."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()

    # 8 total: dataset1(success), train1(success), train2(running),
    #          eval1(failed), loader(success), dataset2(absent stale),
    #          alias(migrated), alias2(migrated)
    assert data["total"] == 8
    assert data["success_count"] == 3
    assert data["failed_count"] == 1
    assert data["running_count"] == 1

    # Check by_result_status
    result_statuses = {s["status"]: s["count"] for s in data["by_result_status"]}
    assert result_statuses.get("success", 0) == 3
    assert result_statuses.get("failed", 0) == 1
    assert result_statuses.get("incomplete", 0) == 1
    assert result_statuses.get("absent", 0) == 1
    assert result_statuses.get("migrated", 0) == 2


def test_combined_filters(client: TestClient, populated_furu_root: Path) -> None:
    """Test combining multiple filters."""
    response = client.get(
        "/api/experiments?result_status=success&namespace=dashboard.pipelines.PrepareDataset"
    )
    assert response.status_code == 200
    data = response.json()
    # dataset1 matches
    assert data["total"] == 1
    assert data["experiments"][0]["result_status"] == "success"
    assert data["experiments"][0]["namespace"].startswith(
        "dashboard.pipelines.PrepareDataset"
    )


def test_experiments_with_dependencies(
    client: TestClient, populated_with_dependencies: Path
) -> None:
    """Test that experiments with real dependencies are properly created."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()

    # Should have 5 experiments all successfully completed
    assert data["total"] == 5
    for exp in data["experiments"]:
        assert exp["result_status"] == "success"

    # Check we have the expected class types
    class_names = {exp["class_name"] for exp in data["experiments"]}
    assert "PrepareDataset" in class_names
    assert "TrainModel" in class_names
    assert "EvalModel" in class_names
    assert "MultiDependencyPipeline" in class_names


# =============================================================================
# Tests for new filtering API endpoints: backend, hostname, user, date range, config
# These tests use the shared populated_furu_root fixture which has:
# - dataset1: success, local, gpu-01, alice, 2025-01-01
# - train1: success, local, gpu-01, alice, 2025-01-02
# - train2: running, submitit, gpu-02, bob, 2025-01-03
# - eval1: failed, local, gpu-02, alice, 2025-01-04
# - loader: success, submitit, gpu-01, bob, 2024-06-01
# - dataset2: absent (stale schema)
# =============================================================================


def test_list_experiments_filter_by_backend(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by backend via API."""
    # Filter by local backend (dataset1, train1, eval1, aliases = 5 experiments)
    response = client.get("/api/experiments?backend=local")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    for exp in data["experiments"]:
        assert exp["backend"] == "local"

    # Filter by submitit backend (train2, loader = 2 experiments)
    response = client.get("/api/experiments?backend=submitit")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    for exp in data["experiments"]:
        assert exp["backend"] == "submitit"


def test_list_experiments_filter_by_hostname(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by hostname via API."""
    # Filter by gpu-01 (dataset1, train1, loader, aliases = 5 experiments)
    response = client.get("/api/experiments?hostname=gpu-01")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    for exp in data["experiments"]:
        assert exp["hostname"] == "gpu-01"

    # Filter by gpu-02 (train2, eval1 = 2 experiments)
    response = client.get("/api/experiments?hostname=gpu-02")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    for exp in data["experiments"]:
        assert exp["hostname"] == "gpu-02"


def test_list_experiments_filter_by_user(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by user via API."""
    # Filter by alice (dataset1, train1, eval1, aliases = 5 experiments)
    response = client.get("/api/experiments?user=alice")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    for exp in data["experiments"]:
        assert exp["user"] == "alice"

    # Filter by bob (train2, loader = 2 experiments)
    response = client.get("/api/experiments?user=bob")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    for exp in data["experiments"]:
        assert exp["user"] == "bob"


def test_list_experiments_filter_by_started_after(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by started_after via API."""
    # Filter for experiments started after 2025-01-01 (train1, train2, eval1 = 3)
    response = client.get("/api/experiments?started_after=2025-01-01T12:00:00%2B00:00")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3


def test_list_experiments_filter_by_started_before(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by started_before via API."""
    # Filter for experiments started before 2025-01-01 (loader = 1)
    response = client.get("/api/experiments?started_before=2025-01-01T00:00:00%2B00:00")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["experiments"][0]["started_at"] == "2024-06-01T10:00:00+00:00"


def test_list_experiments_filter_by_updated_after(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by updated_after via API."""
    # Filter for experiments updated after 2025-01-03 (eval1)
    response = client.get("/api/experiments?updated_after=2025-01-03T12:00:00%2B00:00")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1


def test_list_experiments_filter_by_updated_before(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by updated_before via API."""
    # Filter for experiments updated before 2025-01-01 (loader = 1)
    response = client.get("/api/experiments?updated_before=2025-01-01T00:00:00%2B00:00")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["experiments"][0]["updated_at"] == "2024-06-01T11:00:00+00:00"


def test_list_experiments_filter_by_config_filter(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by config_filter via API."""
    # Filter by config name=mnist (dataset1 + aliases)
    response = client.get("/api/experiments?config_filter=name%3Dmnist")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["experiments"][0]["class_name"] == "PrepareDataset"

    # Filter by config name=cifar (stale schema)
    response = client.get("/api/experiments?config_filter=name%3Dcifar&schema=any")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["experiments"][0]["class_name"] == "PrepareDataset"


def test_list_experiments_filter_by_nested_config_filter(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test filtering experiments by nested config path via API."""
    response = client.get(
        "/api/experiments?config_filter=dataset.name%3Dmnist&schema=any",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert all(exp["class_name"] == "TrainModel" for exp in data["experiments"])


def test_list_experiments_combined_new_filters(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test combining multiple new filters via API."""
    # Combine backend=local + user=alice (dataset1, train1, eval1, aliases = 5)
    response = client.get("/api/experiments?backend=local&user=alice")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5

    # Combine backend=local + hostname=gpu-01 (dataset1, train1, aliases = 4)
    response = client.get("/api/experiments?backend=local&hostname=gpu-01")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 4

    # Combine backend=submitit + hostname=gpu-01 (loader = 1)
    response = client.get("/api/experiments?backend=submitit&hostname=gpu-01")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1

    # Combine result_status=success + user=alice (dataset1, train1 = 2)
    response = client.get("/api/experiments?result_status=success&user=alice")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_list_experiments_new_fields_in_response(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test that new fields are included in response."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()

    # Check structure of returned experiments - all with attempts should have new fields
    for exp in data["experiments"]:
        assert "schema_key" in exp
        assert "current_schema_key" in exp
        assert "is_stale" in exp
        assert "is_alias" in exp
        # Experiments with attempts should have these fields
        if exp["attempt_status"] is not None:
            assert "backend" in exp
            assert "hostname" in exp
            assert "user" in exp


def test_list_experiments_filter_no_match_returns_empty(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test that filters return empty list when no experiments match."""
    # Test each filter type returns empty when no match
    response = client.get("/api/experiments?backend=nonexistent")
    assert response.status_code == 200
    assert response.json()["total"] == 0

    response = client.get("/api/experiments?hostname=nonexistent")
    assert response.status_code == 200
    assert response.json()["total"] == 0

    response = client.get("/api/experiments?user=nonexistent")
    assert response.status_code == 200
    assert response.json()["total"] == 0

    response = client.get("/api/experiments?config_filter=name%3Dnonexistent")
    assert response.status_code == 200
    assert response.json()["total"] == 0

    response = client.get("/api/experiments?hostname=nonexistent")
    assert response.status_code == 200
    assert response.json()["total"] == 0

    response = client.get("/api/experiments?user=nonexistent")
    assert response.status_code == 200
    assert response.json()["total"] == 0

    response = client.get("/api/experiments?config_filter=name%3Dnonexistent")
    assert response.status_code == 200
    assert response.json()["total"] == 0


# =============================================================================
# Tests for query-based POST search endpoint
# =============================================================================


def test_search_experiments_post_result_status_filter(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test searching experiments using a result status query AST."""
    response = client.post(
        "/api/experiments/search",
        json={"query": (Q.exp.result_status == "success").model_dump()},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["experiments"]) == 3
    for exp in data["experiments"]:
        assert exp["result_status"] == "success"


def test_search_experiments_post_nested_config_filter(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test searching by nested config field using a query AST."""
    response = client.post(
        "/api/experiments/search",
        json={"query": (Q.config.dataset.name == "mnist").model_dump()},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert all(exp["class_name"] == "TrainModel" for exp in data["experiments"])


def test_search_experiments_post_type_filter(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test searching experiments using a type filter AST."""
    response = client.post(
        "/api/experiments/search",
        json={"query": (Q.config["__class__"].type_is(TrainModel)).model_dump()},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    for exp in data["experiments"]:
        assert exp["class_name"] == "TrainModel"


def test_search_experiments_post_rejects_invalid_query_payload(
    client: TestClient,
) -> None:
    """Test that invalid query payloads return HTTP 400."""
    query_too_many_nodes = {
        "op": "or",
        "args": [{"op": "eq", "path": "exp.result_status", "value": "success"}] * 201,
    }
    response = client.post(
        "/api/experiments/search",
        json={"query": query_too_many_nodes},
    )
    assert response.status_code == 400
    assert "max allowed is" in response.json()["detail"]


def test_search_experiments_post_rejects_too_deep_query_payload(
    client: TestClient,
) -> None:
    """Test that deeply nested query payloads are rejected."""
    query: dict[str, object] = {
        "op": "eq",
        "path": "exp.result_status",
        "value": "success",
    }
    for _ in range(35):
        query = {"op": "not", "arg": query}

    response = client.post(
        "/api/experiments/search",
        json={"query": query},
    )
    assert response.status_code == 400
    assert "max allowed is" in response.json()["detail"]


# =============================================================================
# Tests for DAG API endpoint
# =============================================================================


def test_dag_endpoint_empty(client: TestClient, temp_furu_root: Path) -> None:
    """Test DAG endpoint with no experiments."""
    response = client.get("/api/dag")
    assert response.status_code == 200
    data = response.json()
    assert data["total_nodes"] == 0
    assert data["total_edges"] == 0
    assert data["total_experiments"] == 0
    assert data["nodes"] == []
    assert data["edges"] == []


def test_dag_endpoint(client: TestClient, populated_furu_root: Path) -> None:
    """Test DAG endpoint with experiments."""
    response = client.get("/api/dag")
    assert response.status_code == 200
    data = response.json()

    # Fixture has 4 unique classes
    assert data["total_nodes"] == 4
    assert data["total_experiments"] == 6

    # Check node structure
    assert len(data["nodes"]) == 4
    node = data["nodes"][0]
    assert "id" in node
    assert "class_name" in node
    assert "full_class_name" in node
    assert "experiments" in node
    assert "total_count" in node
    assert "success_count" in node
    assert "failed_count" in node
    assert "running_count" in node

    # Check edge structure
    assert len(data["edges"]) >= 2
    edge = data["edges"][0]
    assert "source" in edge
    assert "target" in edge
    assert "field_name" in edge


def test_dag_endpoint_node_counts(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test that DAG nodes have correct status counts."""
    response = client.get("/api/dag")
    assert response.status_code == 200
    data = response.json()

    # Find the PrepareDataset node (has 2 experiments: 1 success, 1 absent)
    prepare_node = next(
        (n for n in data["nodes"] if n["class_name"] == "PrepareDataset"), None
    )
    assert prepare_node is not None
    assert prepare_node["total_count"] == 2
    assert prepare_node["success_count"] == 1

    # Find the TrainModel node (has 2 experiments: 1 success, 1 running)
    train_node = next(
        (n for n in data["nodes"] if n["class_name"] == "TrainModel"), None
    )
    assert train_node is not None
    assert train_node["total_count"] == 2
    assert train_node["success_count"] == 1
    assert train_node["running_count"] == 1


def test_dag_endpoint_experiment_details(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test that DAG nodes contain experiment details."""
    response = client.get("/api/dag")
    assert response.status_code == 200
    data = response.json()

    # Find a node with experiments
    node_with_experiments = next(
        (n for n in data["nodes"] if n["total_count"] > 0), None
    )
    assert node_with_experiments is not None
    assert len(node_with_experiments["experiments"]) > 0

    # Check experiment structure
    exp = node_with_experiments["experiments"][0]
    assert "namespace" in exp
    assert "furu_hash" in exp
    assert "result_status" in exp


def test_dag_endpoint_edge_relationships(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test that DAG edges represent correct dependency relationships."""
    response = client.get("/api/dag")
    assert response.status_code == 200
    data = response.json()

    # Find the edge from PrepareDataset to TrainModel
    train_edge = next((e for e in data["edges"] if "TrainModel" in e["target"]), None)
    assert train_edge is not None
    assert "PrepareDataset" in train_edge["source"]
    assert train_edge["field_name"] == "dataset"

    # Find the edge from TrainModel to EvalModel
    eval_edge = next((e for e in data["edges"] if "EvalModel" in e["target"]), None)
    assert eval_edge is not None
    assert "TrainModel" in eval_edge["source"]
    assert eval_edge["field_name"] == "model"


def test_dag_endpoint_with_real_dependencies(
    client: TestClient, populated_with_dependencies: Path
) -> None:
    """Test DAG endpoint with experiments created via get."""
    response = client.get("/api/dag")
    assert response.status_code == 200
    data = response.json()

    # All experiments should be successful
    assert data["total_experiments"] == 5
    for node in data["nodes"]:
        assert node["success_count"] == node["total_count"]


# =============================================================================
# Tests for Relationships API endpoint
# =============================================================================


def test_relationships_endpoint_not_found(
    client: TestClient, temp_furu_root: Path
) -> None:
    """Test relationships endpoint returns 404 for nonexistent experiment."""
    response = client.get("/api/experiments/nonexistent.namespace/abc123/relationships")
    assert response.status_code == 404


def test_relationships_endpoint_no_relationships(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test relationships endpoint for experiment with no parents (root experiment)."""
    # Get the first PrepareDataset experiment (has no parents, may have children)
    list_response = client.get(
        "/api/experiments?namespace=dashboard.pipelines.PrepareDataset"
    )
    assert list_response.status_code == 200
    experiments = list_response.json()["experiments"]
    assert len(experiments) > 0
    exp = experiments[0]

    response = client.get(
        f"/api/experiments/{exp['namespace']}/{exp['furu_hash']}/relationships"
    )
    assert response.status_code == 200
    data = response.json()

    # PrepareDataset has no parents
    assert data["parents"] == []
    # It should have children (TrainModel depends on it)
    assert "children" in data


def test_relationships_endpoint_has_parents(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test relationships endpoint for experiment that has parents."""
    # Get a TrainModel experiment (has PrepareDataset as parent)
    list_response = client.get(
        "/api/experiments?namespace=dashboard.pipelines.TrainModel"
    )
    assert list_response.status_code == 200
    experiments = list_response.json()["experiments"]
    assert len(experiments) > 0
    exp = experiments[0]

    response = client.get(
        f"/api/experiments/{exp['namespace']}/{exp['furu_hash']}/relationships"
    )
    assert response.status_code == 200
    data = response.json()

    # TrainModel has PrepareDataset as parent via "dataset" field
    assert len(data["parents"]) == 1
    parent = data["parents"][0]
    assert parent["field_name"] == "dataset"
    assert parent["class_name"] == "PrepareDataset"
    assert parent["full_class_name"].endswith("PrepareDataset")
    # Should have resolved the parent experiment
    assert parent["namespace"] is not None
    assert parent["furu_hash"] is not None
    assert parent["result_status"] is not None


def test_relationships_endpoint_has_children(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test relationships endpoint for experiment that has children."""
    # Get dataset1 (used by train1 and train2)
    # First get it via the detail endpoint to find its hash
    list_response = client.get(
        "/api/experiments?namespace=dashboard.pipelines.PrepareDataset&config_filter=name%3Dmnist"
    )
    assert list_response.status_code == 200
    experiments = list_response.json()["experiments"]
    assert len(experiments) >= 1
    exp = next(
        experiment
        for experiment in experiments
        if experiment["result_status"] == "success"
    )

    response = client.get(
        f"/api/experiments/{exp['namespace']}/{exp['furu_hash']}/relationships"
    )
    assert response.status_code == 200
    data = response.json()

    # dataset1 is used by both train1 and train2
    assert len(data["children"]) == 2
    for child in data["children"]:
        assert child["class_name"] == "TrainModel"
        assert child["field_name"] == "dataset"
        assert child["namespace"].endswith("TrainModel")
        assert child["furu_hash"] is not None
        assert child["result_status"] in ["success", "incomplete"]


def test_relationships_endpoint_with_real_dependencies(
    client: TestClient, populated_with_dependencies: Path
) -> None:
    """Test relationships endpoint with experiments created via get."""
    # Get the TrainModel experiment
    list_response = client.get(
        "/api/experiments?namespace=dashboard.pipelines.TrainModel"
    )
    assert list_response.status_code == 200
    experiments = list_response.json()["experiments"]
    assert len(experiments) == 1
    train_exp = experiments[0]

    response = client.get(
        f"/api/experiments/{train_exp['namespace']}/{train_exp['furu_hash']}/relationships"
    )
    assert response.status_code == 200
    data = response.json()

    # TrainModel has 1 parent (dataset1)
    assert len(data["parents"]) == 1
    assert data["parents"][0]["field_name"] == "dataset"
    assert data["parents"][0]["namespace"] is not None  # Found the experiment

    # TrainModel has 1 child (EvalModel)
    assert len(data["children"]) == 1
    assert data["children"][0]["class_name"] == "EvalModel"
    assert data["children"][0]["field_name"] == "model"


def test_relationships_parent_structure(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test that parent relationships include all required fields."""
    # Get an EvalModel experiment (has TrainModel as parent)
    list_response = client.get(
        "/api/experiments?namespace=dashboard.pipelines.EvalModel"
    )
    assert list_response.status_code == 200
    experiments = list_response.json()["experiments"]
    assert len(experiments) == 1
    exp = experiments[0]

    response = client.get(
        f"/api/experiments/{exp['namespace']}/{exp['furu_hash']}/relationships"
    )
    assert response.status_code == 200
    data = response.json()

    assert len(data["parents"]) == 1
    parent = data["parents"][0]

    # Check all required fields are present
    assert "field_name" in parent
    assert "class_name" in parent
    assert "full_class_name" in parent
    assert "namespace" in parent
    assert "furu_hash" in parent
    assert "result_status" in parent
    assert "config" in parent

    # Parent should have config data
    assert parent["config"] is not None
    assert isinstance(parent["config"], dict)


def test_relationships_child_structure(
    client: TestClient, populated_furu_root: Path
) -> None:
    """Test that child relationships include all required fields."""
    # Get a TrainModel experiment (has EvalModel as child)
    list_response = client.get(
        "/api/experiments?namespace=dashboard.pipelines.TrainModel&result_status=success"
    )
    assert list_response.status_code == 200
    experiments = list_response.json()["experiments"]
    assert len(experiments) >= 1
    exp = next(
        experiment
        for experiment in experiments
        if experiment["result_status"] == "success"
    )

    response = client.get(
        f"/api/experiments/{exp['namespace']}/{exp['furu_hash']}/relationships"
    )
    assert response.status_code == 200
    data = response.json()

    # Find children (may have EvalModel)
    if len(data["children"]) > 0:
        child = data["children"][0]

        # Check all required fields are present
        assert "field_name" in child
        assert "class_name" in child
        assert "full_class_name" in child
        assert "namespace" in child
        assert "furu_hash" in child
        assert "result_status" in child
