"""Pytest fixtures for dashboard tests.

FIXTURE SELECTION GUIDE
=======================

Use `populated_furu_root` (module-scoped, fast) when:
- Test only reads/queries existing experiments
- Test doesn't need specific isolated data
- Test can work with the shared fixture data (see _create_populated_experiments)

Use `temp_furu_root` (function-scoped, slow) when:
- Test needs an empty directory (e.g., testing empty state)
- Test needs to create experiments with specific attributes not in the shared fixture
- Test mutates experiment state
- Test needs isolated data to verify "no match" scenarios

The populated fixture creates experiments once per test module and reuses them,
which is significantly faster than creating experiments for each test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from furu.config import FURU_CONFIG
from furu.dashboard.main import app
from furu.serialization import FuruSerializer
from furu.storage import (
    MetadataManager,
    MigrationManager,
    MigrationRecord,
    StateManager,
)
from furu.storage.state import _StateResultMigrated

from .pipelines import (
    DataLoader,
    EvalModel,
    MultiDependencyPipeline,
    PrepareDataset,
    TrainModel,
)


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def temp_furu_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Create a temporary Furu root directory and configure it.

    Use this fixture when tests need isolated/empty state or must create
    specific experiments. For read-only tests, prefer `populated_furu_root`.
    """
    monkeypatch.setattr(FURU_CONFIG, "base_root", tmp_path)
    monkeypatch.setattr(
        FURU_CONFIG,
        "version_controlled_root_override",
        tmp_path / "furu-data" / "artifacts",
    )
    monkeypatch.setattr(FURU_CONFIG, "record_git", "ignore")
    monkeypatch.setattr(FURU_CONFIG, "allow_no_git_origin", False)
    monkeypatch.setattr(FURU_CONFIG, "poll_interval", 0.01)
    monkeypatch.setattr(FURU_CONFIG, "stale_timeout", 0.1)
    monkeypatch.setattr(FURU_CONFIG, "max_wait_time_sec", None)
    monkeypatch.setattr(FURU_CONFIG, "lease_duration_sec", 0.05)
    monkeypatch.setattr(FURU_CONFIG, "heartbeat_interval_sec", 0.01)
    monkeypatch.setattr(FURU_CONFIG, "retry_failed", True)

    yield tmp_path


# Module-scoped fixtures for read-only tests (much faster)
@pytest.fixture(scope="module")
def module_furu_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a module-scoped temporary Furu root directory."""
    return tmp_path_factory.mktemp("furu_root")


@pytest.fixture(scope="module")
def _configure_furu_for_module(
    module_furu_root: Path,
) -> Generator[Path, None, None]:
    """Configure FURU_CONFIG for module-scoped tests."""
    # Save original values
    orig_base_root = FURU_CONFIG.base_root
    orig_version_controlled_root_override = FURU_CONFIG.version_controlled_root_override
    orig_record_git = FURU_CONFIG.record_git
    orig_allow_no_git_origin = FURU_CONFIG.allow_no_git_origin
    orig_poll_interval = FURU_CONFIG.poll_interval
    orig_stale_timeout = FURU_CONFIG.stale_timeout
    orig_max_wait = FURU_CONFIG.max_wait_time_sec
    orig_lease_duration = FURU_CONFIG.lease_duration_sec
    orig_heartbeat = FURU_CONFIG.heartbeat_interval_sec
    orig_retry_failed = FURU_CONFIG.retry_failed

    # Set test values
    FURU_CONFIG.base_root = module_furu_root
    FURU_CONFIG.version_controlled_root_override = (
        module_furu_root / "furu-data" / "artifacts"
    )
    FURU_CONFIG.record_git = "ignore"
    FURU_CONFIG.allow_no_git_origin = False
    FURU_CONFIG.poll_interval = 0.01
    FURU_CONFIG.stale_timeout = 0.1
    FURU_CONFIG.max_wait_time_sec = None
    FURU_CONFIG.lease_duration_sec = 0.05
    FURU_CONFIG.heartbeat_interval_sec = 0.01
    FURU_CONFIG.retry_failed = True

    yield module_furu_root

    # Restore original values
    FURU_CONFIG.base_root = orig_base_root
    FURU_CONFIG.version_controlled_root_override = orig_version_controlled_root_override
    FURU_CONFIG.record_git = orig_record_git
    FURU_CONFIG.allow_no_git_origin = orig_allow_no_git_origin
    FURU_CONFIG.poll_interval = orig_poll_interval
    FURU_CONFIG.stale_timeout = orig_stale_timeout
    FURU_CONFIG.max_wait_time_sec = orig_max_wait
    FURU_CONFIG.lease_duration_sec = orig_lease_duration
    FURU_CONFIG.heartbeat_interval_sec = orig_heartbeat
    FURU_CONFIG.retry_failed = orig_retry_failed


def create_experiment_from_furu(
    furu_obj: object,
    result_status: str = "success",
    attempt_status: str | None = None,
    backend: str = "local",
    hostname: str = "test-host",
    user: str = "testuser",
    started_at: str = "2025-01-01T11:00:00+00:00",
    updated_at: str = "2025-01-01T12:00:00+00:00",
) -> Path:
    """
    Create an experiment directory from an actual Furu object.

    This creates realistic metadata and state by using the actual Furu
    serialization and metadata systems.

    Args:
        furu_obj: A Furu subclass instance
        result_status: One of: absent, incomplete, success, failed
        attempt_status: Optional attempt status (queued, running, success, failed, etc.)
        backend: Backend type (local, submitit)
        hostname: Hostname where the experiment ran
        user: User who ran the experiment
        started_at: ISO timestamp for when the experiment started
        updated_at: ISO timestamp for when the experiment was last updated

    Returns:
        Path to the created experiment directory
    """
    # Get the furu_dir from the object (uses real path computation)
    directory = furu_obj.furu_dir  # type: ignore[attr-defined]
    StateManager.ensure_internal_dir(directory)

    # Create metadata using the actual metadata system
    metadata = MetadataManager.create_metadata(
        furu_obj,  # type: ignore[arg-type]
        directory,
        ignore_diff=True,
    )
    MetadataManager.write_metadata(metadata, directory)

    # Build state based on result_status
    if result_status == "absent":
        result: dict[str, str] = {"status": "absent"}
    elif result_status == "incomplete":
        result = {"status": "incomplete"}
    elif result_status == "success":
        result = {"status": "success", "created_at": updated_at}
    else:  # failed
        result = {"status": "failed"}

    # Build attempt if status provided
    attempt: dict[str, str | int | float | dict[str, str | int] | None] | None = None
    if attempt_status:
        attempt = {
            "id": f"attempt-{FuruSerializer.compute_hash(furu_obj)[:8]}",
            "number": 1,
            "backend": backend,
            "status": attempt_status,
            "started_at": started_at,
            "lease_duration_sec": 120.0,
            "lease_expires_at": "2025-01-01T13:00:00+00:00",
            "owner": {
                "pid": 12345,
                "host": hostname,
                "hostname": hostname,
                "user": user,
            },
            "scheduler": {},
        }
        if attempt_status in ("success", "failed", "crashed", "cancelled", "preempted"):
            attempt["ended_at"] = updated_at
        if attempt_status == "failed":
            attempt["error"] = {
                "type": "RuntimeError",
                "message": "Test error",
            }

    state = {
        "schema_version": 1,
        "result": result,
        "attempt": attempt,
        "updated_at": updated_at,
    }

    state_path = StateManager.get_state_path(directory)
    state_path.write_text(json.dumps(state, indent=2))

    # Write success marker if successful
    if result_status == "success":
        success_marker = StateManager.get_success_marker_path(directory)
        success_marker.write_text(
            json.dumps(
                {
                    "attempt_id": attempt["id"] if attempt else "unknown",
                    "created_at": updated_at,
                }
            )
        )

    return directory


def _create_populated_experiments(root: Path) -> None:
    """
    Seed the given root directory with a set of realistic, interdependent experiment records for dashboard/filter tests.
    
    Creates on-disk experiment metadata and state (including success, running, failed, and absent states), builds dependency relationships between experiments (PrepareDataset -> TrainModel -> EvalModel), adds experiments in a different namespace, creates one dataset with an absent/stale schema, and writes two migrated "alias" migration records pointing back to the first dataset. The function mutates the filesystem under `root` and returns None.
    """
    # Create a base dataset (successful, local, gpu-01, alice, early 2025)
    dataset1 = PrepareDataset(name="mnist", version="v1")
    create_experiment_from_furu(
        dataset1,
        result_status="success",
        attempt_status="success",
        backend="local",
        hostname="gpu-01",
        user="alice",
        started_at="2025-01-01T10:00:00+00:00",
        updated_at="2025-01-01T11:00:00+00:00",
    )

    # Create a training run that depends on the dataset (successful, local, gpu-01, alice)
    train1 = TrainModel(lr=0.001, steps=1000, dataset=dataset1)
    create_experiment_from_furu(
        train1,
        result_status="success",
        attempt_status="success",
        backend="local",
        hostname="gpu-01",
        user="alice",
        started_at="2025-01-02T10:00:00+00:00",
        updated_at="2025-01-02T11:00:00+00:00",
    )

    # Create another training run with different params (running, submitit, gpu-02, bob)
    train2 = TrainModel(lr=0.0001, steps=2000, dataset=dataset1)
    create_experiment_from_furu(
        train2,
        result_status="incomplete",
        attempt_status="running",
        backend="submitit",
        hostname="gpu-02",
        user="bob",
        started_at="2025-01-03T10:00:00+00:00",
        updated_at="2025-01-03T11:00:00+00:00",
    )

    # Create an evaluation that depends on training (failed, local, gpu-02, alice)
    eval1 = EvalModel(model=train1, eval_split="test")
    create_experiment_from_furu(
        eval1,
        result_status="failed",
        attempt_status="failed",
        backend="local",
        hostname="gpu-02",
        user="alice",
        started_at="2025-01-04T10:00:00+00:00",
        updated_at="2025-01-04T11:00:00+00:00",
    )

    # Create a data loader in a different namespace (successful, submitit, gpu-01, bob, 2024)
    loader = DataLoader(source="s3", format="parquet")
    create_experiment_from_furu(
        loader,
        result_status="success",
        attempt_status="success",
        backend="submitit",
        hostname="gpu-01",
        user="bob",
        started_at="2024-06-01T10:00:00+00:00",
        updated_at="2024-06-01T11:00:00+00:00",
    )

    # Create another dataset with absent status (no attempt)
    dataset2 = PrepareDataset(name="cifar", version="v2")
    dataset2_dir = create_experiment_from_furu(
        dataset2, result_status="absent", attempt_status=None
    )
    dataset2_metadata_path = MetadataManager.get_metadata_path(dataset2_dir)
    dataset2_metadata = json.loads(dataset2_metadata_path.read_text())
    dataset2_metadata["schema_key"] = ["name"]
    dataset2_metadata_path.write_text(json.dumps(dataset2_metadata, indent=2))

    def set_alias_state(state) -> None:
        """
        Mark the provided experiment state as migrated and clear any active attempt.
        
        Parameters:
            state: The experiment state object to modify; its `result` will be set to a migrated result and its `attempt` cleared.
        """
        state.result = _StateResultMigrated(status="migrated")
        state.attempt = None

    # Create an alias dataset that points back to dataset1
    dataset_alias = PrepareDataset(name="mnist", version="v2")
    alias_dir = dataset_alias.furu_dir
    StateManager.ensure_internal_dir(alias_dir)
    MetadataManager.write_metadata(
        MetadataManager.create_metadata(dataset_alias, alias_dir, ignore_diff=True),
        alias_dir,
    )

    StateManager.update_state(alias_dir, set_alias_state)
    alias_record = MigrationRecord(
        kind="alias",
        policy="alias",
        from_namespace="dashboard.pipelines.PrepareDataset",
        from_hash=FuruSerializer.compute_hash(dataset1),
        from_root="data",
        to_namespace="dashboard.pipelines.PrepareDataset",
        to_hash=FuruSerializer.compute_hash(dataset_alias),
        to_root="data",
        migrated_at="2025-01-05T10:00:00+00:00",
        overwritten_at=None,
        default_values={"language": "spanish"},
        origin="tests",
        note="alias fixture",
    )
    MigrationManager.write_migration(alias_record, alias_dir)

    dataset_alias_second = PrepareDataset(name="mnist", version="v4")
    alias_second_dir = dataset_alias_second.furu_dir
    StateManager.ensure_internal_dir(alias_second_dir)
    MetadataManager.write_metadata(
        MetadataManager.create_metadata(
            dataset_alias_second,
            alias_second_dir,
            ignore_diff=True,
        ),
        alias_second_dir,
    )
    StateManager.update_state(alias_second_dir, set_alias_state)
    alias_second_record = MigrationRecord(
        kind="alias",
        policy="alias",
        from_namespace="dashboard.pipelines.PrepareDataset",
        from_hash=FuruSerializer.compute_hash(dataset1),
        from_root="data",
        to_namespace="dashboard.pipelines.PrepareDataset",
        to_hash=FuruSerializer.compute_hash(dataset_alias_second),
        to_root="data",
        migrated_at="2025-01-05T12:00:00+00:00",
        overwritten_at=None,
        default_values={"language": "french"},
        origin="tests",
        note="alias fixture 2",
    )
    MigrationManager.write_migration(alias_second_record, alias_second_dir)


@pytest.fixture(scope="module")
def populated_furu_root(_configure_furu_for_module: Path) -> Path:
    """
    Creates a module-scoped Furu root populated with sample experiments.
    
    Experiments are created once for the test module and reused across tests to speed up read-only test runs. See _create_populated_experiments() for the exact fixtures seeded.
    
    Returns:
        Path: Path to the configured Furu root directory containing the sample experiments.
    """
    root = _configure_furu_for_module
    _create_populated_experiments(root)
    return root


@pytest.fixture
def populated_with_dependencies(temp_furu_root: Path) -> Path:
    """Create experiments with a full dependency chain.

    This fixture actually runs get() to create real experiments,
    so it must be function-scoped.

    This creates a realistic DAG:
    - dataset1 (PrepareDataset)
    - dataset2 (PrepareDataset)
    - train (TrainModel) depends on dataset1
    - eval (EvalModel) depends on train
    - multi (MultiDependencyPipeline) depends on dataset1 and dataset2
    """
    # Base datasets
    dataset1 = PrepareDataset(name="train_data", version="v1")
    dataset1.get()

    dataset2 = PrepareDataset(name="val_data", version="v1")
    dataset2.get()

    # Training depends on dataset1
    train = TrainModel(lr=0.001, steps=500, dataset=dataset1)
    train.get()

    # Evaluation depends on training
    eval_model = EvalModel(model=train, eval_split="validation")
    eval_model.get()

    # Multi-dependency pipeline
    multi = MultiDependencyPipeline(
        dataset1=dataset1, dataset2=dataset2, output_name="merged"
    )
    multi.get()

    return temp_furu_root