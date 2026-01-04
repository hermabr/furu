#!/usr/bin/env python3
"""Generate realistic test data for e2e tests.

This script creates actual Huldra experiments with dependencies using
the Huldra framework. It creates a realistic set of experiments with
various states (success, failed, running) and dependency chains.

Usage:
    python generate_data.py [--clean]

The --clean flag will remove existing data-huldra directory before generating.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add src and examples to path so we can import huldra and the pipelines
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from huldra.config import HULDRA_CONFIG
from huldra.serialization import HuldraSerializer
from huldra.storage import MetadataManager, StateManager

# Import from the examples module which has proper module paths
from my_project.pipelines import PrepareDataset, TrainModel, TrainTextModel  # type: ignore[import-not-found]


def create_mock_experiment(
    huldra_obj: object,
    result_status: str = "success",
    attempt_status: str | None = None,
) -> Path:
    """Create a mock experiment with specified state (without actually running _create)."""
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


def generate_test_data(data_root: Path) -> None:
    """Generate test data for e2e tests."""
    print(f"Generating test data in {data_root}")

    # Configure Huldra to use our data root
    HULDRA_CONFIG.base_root = data_root
    HULDRA_CONFIG.ignore_git_diff = True

    # Create experiments with various states

    # 1. Successful experiments with dependencies (actually run them)
    print("Creating successful experiments with dependencies...")

    # Dataset 1: default toy dataset
    dataset_toy = PrepareDataset(name="toy")
    dataset_toy.load_or_create()
    print(f"  Created: {dataset_toy.__class__.__name__} (toy)")

    # Dataset 2: MNIST dataset
    dataset_mnist = PrepareDataset(name="mnist")
    dataset_mnist.load_or_create()
    print(f"  Created: {dataset_mnist.__class__.__name__} (mnist)")

    # Training model on toy dataset
    train_toy = TrainModel(lr=0.001, steps=1000, dataset=dataset_toy)
    train_toy.load_or_create()
    print(f"  Created: {train_toy.__class__.__name__} (toy, lr=0.001)")

    # Text model training
    text_model = TrainTextModel(dataset=dataset_toy)
    text_model.load_or_create()
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
    text_model2.load_or_create()
    print(f"  Created: {text_model2.__class__.__name__} (mnist)")

    # Additional training run
    train_extra = TrainModel(lr=0.005, steps=2000, dataset=dataset_toy)
    train_extra.load_or_create()
    print(f"  Created: {train_extra.__class__.__name__} (toy, lr=0.005)")

    print("\nGenerated 10 experiments total")
    print("  - 6 successful")
    print("  - 1 running")
    print("  - 1 failed")
    print("  - 1 queued")
    print("  - 1 absent")


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
        default=Path(__file__).parent.parent / "data-huldra",
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
