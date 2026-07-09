from pathlib import Path


def data_dir_in(base_dir: Path) -> Path:
    return base_dir / "data"


def result_dir_in(base_dir: Path) -> Path:
    return base_dir / "result"


def result_manifest_path_in(base_dir: Path) -> Path:
    return result_dir_in(base_dir) / "manifest.json"


def result_manifest_overlay_path_in(base_dir: Path) -> Path:
    return base_dir / "result-manifest.json"


def metadata_path_in(base_dir: Path) -> Path:
    return base_dir / "metadata.json"


def provenance_path_in(base_dir: Path) -> Path:
    return base_dir / "provenance.json"


def schema_snapshot_path_in(base_dir: Path) -> Path:
    # base_dir is {fqn}/{schema_hash}/{artifact_hash}; the snapshot is written
    # once per (class, schema-hash), beside the artifact directories.
    return schema_snapshot_path_in_schema_directory(base_dir.parent)


def schema_snapshot_path_in_schema_directory(schema_directory: Path) -> Path:
    return schema_directory / "schema.json"


def run_log_path_in(base_dir: Path) -> Path:
    return base_dir / "run.log"


def execution_coordinator_log_path_in(executor_dir: Path) -> Path:
    return executor_dir / "execution_coordinator.log"


def compute_lock_path_in(base_dir: Path) -> Path:
    return base_dir / "compute.lock"


def result_link_path_in(base_dir: Path) -> Path:
    return base_dir / "result-link.json"
