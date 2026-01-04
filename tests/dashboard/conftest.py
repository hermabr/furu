"""Pytest fixtures for dashboard tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from huldra.config import HULDRA_CONFIG
from huldra.dashboard.main import app
from huldra.serialization import HuldraSerializer
from huldra.storage import MetadataManager, StateManager

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
def temp_huldra_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Create a temporary Huldra root directory and configure it."""
    monkeypatch.setattr(HULDRA_CONFIG, "base_root", tmp_path)
    monkeypatch.setattr(HULDRA_CONFIG, "ignore_git_diff", True)
    monkeypatch.setattr(HULDRA_CONFIG, "poll_interval", 0.01)
    monkeypatch.setattr(HULDRA_CONFIG, "stale_timeout", 0.1)
    monkeypatch.setattr(HULDRA_CONFIG, "lease_duration_sec", 0.05)
    monkeypatch.setattr(HULDRA_CONFIG, "heartbeat_interval_sec", 0.01)

    yield tmp_path


# Module-scoped fixtures for read-only tests (much faster)
@pytest.fixture(scope="module")
def module_huldra_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a module-scoped temporary Huldra root directory."""
    return tmp_path_factory.mktemp("huldra_root")


@pytest.fixture(scope="module")
def _configure_huldra_for_module(
    module_huldra_root: Path,
) -> Generator[Path, None, None]:
    """Configure HULDRA_CONFIG for module-scoped tests."""
    # Save original values
    orig_base_root = HULDRA_CONFIG.base_root
    orig_ignore_git_diff = HULDRA_CONFIG.ignore_git_diff
    orig_poll_interval = HULDRA_CONFIG.poll_interval
    orig_stale_timeout = HULDRA_CONFIG.stale_timeout
    orig_lease_duration = HULDRA_CONFIG.lease_duration_sec
    orig_heartbeat = HULDRA_CONFIG.heartbeat_interval_sec

    # Set test values
    HULDRA_CONFIG.base_root = module_huldra_root
    HULDRA_CONFIG.ignore_git_diff = True
    HULDRA_CONFIG.poll_interval = 0.01
    HULDRA_CONFIG.stale_timeout = 0.1
    HULDRA_CONFIG.lease_duration_sec = 0.05
    HULDRA_CONFIG.heartbeat_interval_sec = 0.01

    yield module_huldra_root

    # Restore original values
    HULDRA_CONFIG.base_root = orig_base_root
    HULDRA_CONFIG.ignore_git_diff = orig_ignore_git_diff
    HULDRA_CONFIG.poll_interval = orig_poll_interval
    HULDRA_CONFIG.stale_timeout = orig_stale_timeout
    HULDRA_CONFIG.lease_duration_sec = orig_lease_duration
    HULDRA_CONFIG.heartbeat_interval_sec = orig_heartbeat


def create_experiment_from_huldra(
    huldra_obj: object,
    result_status: str = "success",
    attempt_status: str | None = None,
) -> Path:
    """
    Create an experiment directory from an actual Huldra object.

    This creates realistic metadata and state by using the actual Huldra
    serialization and metadata systems.

    Args:
        huldra_obj: A Huldra subclass instance
        result_status: One of: absent, incomplete, success, failed
        attempt_status: Optional attempt status (queued, running, success, failed, etc.)

    Returns:
        Path to the created experiment directory
    """
    # Get the huldra_dir from the object (uses real path computation)
    directory = huldra_obj.huldra_dir  # type: ignore[attr-defined]
    directory.mkdir(parents=True, exist_ok=True)

    # Create metadata using the actual metadata system
    metadata = MetadataManager.create_metadata(
        huldra_obj,  # type: ignore[arg-type]
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
        result = {"status": "success", "created_at": "2025-01-01T12:00:00+00:00"}
    else:  # failed
        result = {"status": "failed"}

    # Build attempt if status provided
    attempt: dict[str, str | int | float | dict[str, str | int] | None] | None = None
    if attempt_status:
        attempt = {
            "id": f"attempt-{HuldraSerializer.compute_hash(huldra_obj)[:8]}",
            "number": 1,
            "backend": "local",
            "status": attempt_status,
            "started_at": "2025-01-01T11:00:00+00:00",
            "heartbeat_at": "2025-01-01T11:30:00+00:00",
            "lease_duration_sec": 120.0,
            "lease_expires_at": "2025-01-01T13:00:00+00:00",
            "owner": {
                "pid": 12345,
                "host": "test-host",
                "user": "testuser",
            },
            "scheduler": {},
        }
        if attempt_status in ("success", "failed", "crashed", "cancelled", "preempted"):
            attempt["ended_at"] = "2025-01-01T12:00:00+00:00"
        if attempt_status == "failed":
            attempt["error"] = {
                "type": "RuntimeError",
                "message": "Test error",
            }

    state = {
        "schema_version": 1,
        "result": result,
        "attempt": attempt,
        "updated_at": "2025-01-01T12:00:00+00:00",
    }

    state_path = StateManager.get_state_path(directory)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))

    # Write success marker if successful
    if result_status == "success":
        success_marker = StateManager.get_success_marker_path(directory)
        success_marker.write_text(
            json.dumps(
                {
                    "attempt_id": attempt["id"] if attempt else "unknown",
                    "created_at": "2025-01-01T12:00:00+00:00",
                }
            )
        )

    return directory


def _create_populated_experiments(root: Path) -> None:
    """Create sample experiments in the given root directory.

    Creates experiments with realistic dependencies:
    - PrepareDataset (success) - base dataset
    - TrainModel with dependency on PrepareDataset (success)
    - TrainModel with different params (running)
    - EvalModel that depends on TrainModel (failed)
    - DataLoader in different namespace (success)
    - PrepareDataset with different params (absent)
    """
    # Create a base dataset (successful)
    dataset1 = PrepareDataset(name="mnist", version="v1")
    create_experiment_from_huldra(
        dataset1, result_status="success", attempt_status="success"
    )

    # Create a training run that depends on the dataset (successful)
    train1 = TrainModel(lr=0.001, steps=1000, dataset=dataset1)
    create_experiment_from_huldra(
        train1, result_status="success", attempt_status="success"
    )

    # Create another training run with different params (running)
    train2 = TrainModel(lr=0.0001, steps=2000, dataset=dataset1)
    create_experiment_from_huldra(
        train2, result_status="incomplete", attempt_status="running"
    )

    # Create an evaluation that depends on training (failed)
    eval1 = EvalModel(model=train1, eval_split="test")
    create_experiment_from_huldra(
        eval1, result_status="failed", attempt_status="failed"
    )

    # Create a data loader in a different namespace (successful)
    loader = DataLoader(source="s3", format="parquet")
    create_experiment_from_huldra(
        loader, result_status="success", attempt_status="success"
    )

    # Create another dataset with absent status
    dataset2 = PrepareDataset(name="cifar", version="v2")
    create_experiment_from_huldra(dataset2, result_status="absent", attempt_status=None)


@pytest.fixture(scope="module")
def populated_huldra_root(_configure_huldra_for_module: Path) -> Path:
    """Create a module-scoped Huldra root with sample experiments.

    This is much faster than creating experiments for each test since
    the experiments are created once per module and reused.
    """
    root = _configure_huldra_for_module
    _create_populated_experiments(root)
    return root


@pytest.fixture
def populated_with_dependencies(temp_huldra_root: Path) -> Path:
    """Create experiments with a full dependency chain.

    This fixture actually runs load_or_create() to create real experiments,
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
    dataset1.load_or_create()

    dataset2 = PrepareDataset(name="val_data", version="v1")
    dataset2.load_or_create()

    # Training depends on dataset1
    train = TrainModel(lr=0.001, steps=500, dataset=dataset1)
    train.load_or_create()

    # Evaluation depends on training
    eval_model = EvalModel(model=train, eval_split="validation")
    eval_model.load_or_create()

    # Multi-dependency pipeline
    multi = MultiDependencyPipeline(
        dataset1=dataset1, dataset2=dataset2, output_name="merged"
    )
    multi.load_or_create()

    return temp_huldra_root
