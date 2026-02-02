#!/usr/bin/env python3
"""Generate realistic test data for e2e tests.

This script creates actual Furu experiments with dependencies using
the Furu framework. It creates a realistic set of experiments with
various states (success, failed, running) and dependency chains.

Usage:
    python generate_data.py [--clean]

The --clean flag will remove existing data-furu directory before generating.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add src and examples to path so we can import furu and the pipelines
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

# Import from the examples module which has proper module paths
from my_project.pipelines import (  # type: ignore[import-not-found]
    PrepareDataset,
    TrainModel,
    TrainTextModel,
)

from furu.config import FURU_CONFIG
from furu.serialization import FuruSerializer
from furu.storage import MetadataManager, StateManager


def _create_migrated_aliases(
    dataset_mnist: PrepareDataset,
    dataset_toy: PrepareDataset,
) -> None:
    """
    Create migrated alias records for two dataset pipelines.
    
    This records migrated aliases named "mnist-v2" and "toy-v2" for the provided dataset instances so they are available under those new names in test data.
    
    Parameters:
    	dataset_mnist (PrepareDataset): Source PrepareDataset instance to migrate to the "mnist-v2" alias.
    	dataset_toy (PrepareDataset): Source PrepareDataset instance to migrate to the "toy-v2" alias.
    """
    PrepareDataset.migrate_one(
        from_hash=dataset_mnist.furu_hash,
        from_namespace="my_project.pipelines.PrepareDataset",
        drop_field=("name",),
        set_field={"name": "mnist-v2"},
        origin="e2e",
        note="migration fixture",
    )
    PrepareDataset.migrate_one(
        from_hash=dataset_toy.furu_hash,
        from_namespace="my_project.pipelines.PrepareDataset",
        drop_field=("name",),
        set_field={"name": "toy-v2"},
        origin="e2e",
        note="migration fixture 2",
    )


def create_mock_experiment(
    furu_obj: object,
    result_status: str = "success",
    attempt_status: str | None = None,
) -> Path:
    """
    Create or update on-disk metadata and state files to simulate an experiment run without executing the experiment.
    
    Parameters:
        furu_obj (object): Furu pipeline object whose furu_dir will be used as the experiment directory.
        result_status (str): Desired result state written to the experiment state. One of:
            - "absent": marks the result as absent
            - "incomplete": marks the result as incomplete
            - "success": marks the result as successful (writes a success marker)
            - "failed": marks the result as failed
        attempt_status (str | None): If provided, creates an attempt record with this status (e.g., "running", "success", "failed", "queued", "crashed", "cancelled", "preempted"). When the status indicates completion ("success", "failed", "crashed", "cancelled", "preempted"), an ended_at timestamp is included; when "failed", an error object is added.
    
    Returns:
        Path: The experiment directory path used to write metadata and state files.
    """
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
        result = {"status": "success", "created_at": "2025-01-01T12:00:00+00:00"}
    else:  # failed
        result = {"status": "failed"}

    # Build attempt if status provided
    attempt: dict[str, str | int | float | dict[str, str | int] | None] | None = None
    if attempt_status:
        attempt = {
            "id": f"attempt-{FuruSerializer.compute_hash(furu_obj)[:8]}",
            "number": 1,
            "backend": "local",
            "status": attempt_status,
            "started_at": "2025-01-01T11:00:00+00:00",
            "lease_duration_sec": 120.0,
            "lease_expires_at": "2025-01-01T13:00:00+00:00",
            "owner": {
                "pid": 12345,
                "host": "e2e-test-host",
                "user": "e2e-tester",
            },
            "scheduler": {},
        }
        if attempt_status in ("success", "failed", "crashed", "cancelled", "preempted"):
            attempt["ended_at"] = "2025-01-01T12:00:00+00:00"
        if attempt_status == "failed":
            attempt["error"] = {
                "type": "RuntimeError",
                "message": "Test error for e2e testing",
            }

    state = {
        "schema_version": 1,
        "result": result,
        "attempt": attempt,
        "updated_at": "2025-01-01T12:00:00+00:00",
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
                    "created_at": "2025-01-01T12:00:00+00:00",
                }
            )
        )

    return directory


def generate_test_data(data_root: Path) -> None:
    """
    Generate realistic end-to-end test data and experiment records under the specified root.
    
    Configures Furu to use data_root, creates several real experiments (datasets and model trainings),
    and writes a variety of mock experiment states (running, failed, queued, absent, and migrated aliases)
    to populate a comprehensive test dataset.
    
    Parameters:
        data_root (Path): Target root directory where test data and experiment state will be created.
    """
    print(f"Generating test data in {data_root}")

    # Configure Furu to use our data root
    FURU_CONFIG.base_root = data_root
    FURU_CONFIG.record_git = "ignore"

    # Create experiments with various states

    # 1. Successful experiments with dependencies (actually run them)
    print("Creating successful experiments with dependencies...")

    # Dataset 1: default toy dataset
    dataset_toy = PrepareDataset(name="toy")
    dataset_toy.get()
    print(f"  Created: {dataset_toy.__class__.__name__} (toy)")

    # Dataset 2: MNIST dataset
    dataset_mnist = PrepareDataset(name="mnist")
    dataset_mnist.get()
    print(f"  Created: {dataset_mnist.__class__.__name__} (mnist)")

    # Migrated alias dataset (mnist -> mnist-v2)
    _create_migrated_aliases(dataset_mnist, dataset_toy)
    print("  Created: PrepareDataset aliases (mnist-v2, toy-v2)")

    # Training model on toy dataset
    train_toy = TrainModel(lr=0.001, steps=1000, dataset=dataset_toy)
    train_toy.get()
    print(f"  Created: {train_toy.__class__.__name__} (toy, lr=0.001)")

    # Text model training
    text_model = TrainTextModel(dataset=dataset_toy)
    text_model.get()
    print(f"  Created: {text_model.__class__.__name__}")

    # 2. Mock experiments with different states (don't run _create)
    print("Creating mock experiments with various states...")

    # Running training experiment
    train_running = TrainModel(lr=0.0001, steps=5000, dataset=dataset_mnist)
    create_mock_experiment(
        train_running, result_status="incomplete", attempt_status="running"
    )
    print(f"  Created: {train_running.__class__.__name__} (running)")

    # Failed experiment
    train_failed = TrainModel(lr=0.1, steps=100, dataset=dataset_toy)
    create_mock_experiment(
        train_failed, result_status="failed", attempt_status="failed"
    )
    print(f"  Created: {train_failed.__class__.__name__} (failed)")

    # Queued experiment
    train_queued = TrainModel(lr=0.01, steps=500, dataset=dataset_mnist)
    create_mock_experiment(
        train_queued, result_status="incomplete", attempt_status="queued"
    )
    print(f"  Created: {train_queued.__class__.__name__} (queued)")

    # Absent experiment (no attempt yet)
    dataset_absent = PrepareDataset(name="imagenet")
    create_mock_experiment(dataset_absent, result_status="absent", attempt_status=None)
    print(f"  Created: {dataset_absent.__class__.__name__} (absent)")

    # Another successful text model with different params
    text_model2 = TrainTextModel(dataset=dataset_mnist)
    text_model2.get()
    print(f"  Created: {text_model2.__class__.__name__} (mnist)")

    # Additional training run
    train_extra = TrainModel(lr=0.005, steps=2000, dataset=dataset_toy)
    train_extra.get()
    print(f"  Created: {train_extra.__class__.__name__} (toy, lr=0.005)")

    print("\nGenerated 12 experiments total")
    print("  - 6 successful")
    print("  - 1 running")
    print("  - 1 failed")
    print("  - 1 queued")
    print("  - 1 absent")
    print("  - 2 migrated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate e2e test data")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing data directory before generating",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data-furu",
        help="Directory to store generated data",
    )
    args = parser.parse_args()

    data_root = args.data_dir.resolve()

    if args.clean and data_root.exists():
        print(f"Removing existing data directory: {data_root}")
        shutil.rmtree(data_root)

    data_root.mkdir(parents=True, exist_ok=True)
    generate_test_data(data_root)
    print(f"\nData generated successfully in: {data_root}")


if __name__ == "__main__":
    main()