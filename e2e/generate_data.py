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

import furu
from furu.config import FURU_CONFIG
from furu.serialization import FuruSerializer
from furu.storage import (
    MetadataManager,
    MigrationManager,
    MigrationRecord,
    StateManager,
)

# Import from the examples module which has proper module paths
from my_project.pipelines import PrepareDataset, TrainModel, TrainTextModel  # type: ignore[import-not-found]


def _create_migrated_aliases() -> PrepareDataset:
    dataset_old = PrepareDataset(name="mnist")
    dataset_old.load_or_create()

    def find_candidate(default_values: dict[str, str]):
        candidates = furu.find_migration_candidates(
            namespace=furu.NamespacePair(
                from_namespace="my_project.pipelines.PrepareDataset",
                to_namespace="my_project.pipelines.PrepareDataset",
            ),
            default_values=default_values,
            drop_fields=["name"],
        )
        target_hash = dataset_old._furu_hash
        candidates = [
            candidate
            for candidate in candidates
            if candidate.from_ref.furu_hash == target_hash
        ]
        if not candidates:
            raise ValueError("migration: no candidates found for target class")
        if len(candidates) != 1:
            raise ValueError("migration: expected exactly one candidate")
        return candidates[0]

    furu.apply_migration(
        find_candidate({"name": "mnist-v2"}),
        policy="alias",
        cascade=True,
        origin="e2e",
        note="migration fixture",
        conflict="throw",
    )
    furu.apply_migration(
        find_candidate({"name": "mnist-v3"}),
        policy="alias",
        cascade=True,
        origin="e2e",
        note="migration fixture 2",
        conflict="throw",
    )
    return dataset_old


def create_mock_experiment(
    furu_obj: object,
    result_status: str = "success",
    attempt_status: str | None = None,
) -> Path:
    """Create a mock experiment with specified state (without actually running _create)."""
    directory = furu_obj.furu_dir  # type: ignore[attr-defined]
    directory.mkdir(parents=True, exist_ok=True)

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

    # Configure Furu to use our data root
    FURU_CONFIG.base_root = data_root
    FURU_CONFIG.ignore_git_diff = True

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

    # Migrated alias dataset (mnist -> mnist-v2)
    dataset_old = _create_migrated_aliases()
    print("  Created: PrepareDataset aliases (mnist-v2, mnist-v3)")

    candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="my_project.pipelines.PrepareDataset",
            to_namespace="my_project.pipelines.PrepareDataset",
        ),
        default_values={"name": "mnist-copy"},
        drop_fields=["name"],
    )
    copy_candidates = [
        candidate
        for candidate in candidates
        if candidate.from_ref.furu_hash == dataset_old._furu_hash
    ]
    if not copy_candidates:
        raise ValueError("migration: no copy candidates found")
    furu.apply_migration(
        copy_candidates[0],
        policy="copy",
        cascade=True,
        origin="e2e",
        note="migration copy fixture",
        conflict="throw",
    )

    move_candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="my_project.pipelines.PrepareDataset",
            to_namespace="my_project.pipelines.PrepareDataset",
        ),
        default_values={"name": "mnist-move"},
        drop_fields=["name"],
    )
    move_candidates = [
        candidate
        for candidate in move_candidates
        if candidate.from_ref.furu_hash == dataset_old._furu_hash
    ]
    if not move_candidates:
        raise ValueError("migration: no move candidates found")
    move_candidate = move_candidates[0]
    move_records = furu.apply_migration(
        move_candidate,
        policy="move",
        cascade=True,
        origin="e2e",
        note="migration move fixture",
        conflict="throw",
    )
    move_dir = move_candidate.to_ref.directory
    original_dir = move_candidate.from_ref.directory
    if move_records:
        primary_record = move_records[0]
        if isinstance(primary_record, MigrationRecord):
            original_record = MigrationRecord(
                kind="migrated",
                policy=primary_record.policy,
                from_namespace=primary_record.from_namespace,
                from_hash=primary_record.from_hash,
                from_root=primary_record.from_root,
                to_namespace=primary_record.to_namespace,
                to_hash=primary_record.to_hash,
                to_root=primary_record.to_root,
                migrated_at=primary_record.migrated_at,
                overwritten_at=primary_record.overwritten_at,
                default_values=primary_record.default_values,
                origin=primary_record.origin,
                note=primary_record.note,
            )
            MigrationManager.write_migration(primary_record, move_dir)
            MigrationManager.write_migration(original_record, original_dir)

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

    print("\nGenerated 14 experiments total")
    print("  - 8 successful")
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
