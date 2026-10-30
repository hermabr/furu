from pathlib import Path

# An identity directory {storage}/{fqn}/{schema_hash}/{artifact_hash} holds
# spec.json, compute.lock, run.log (every attempt's log), the in-progress
# attempt/ and one published version directory per code version (v-<hash>,
# v-fixed or v-fixed-<hash>). attempt/ and every version directory share the
# run-directory layout below.


def spec_path_in(identity_dir: Path) -> Path:
    return identity_dir / "spec.json"


def attempt_dir_in(identity_dir: Path) -> Path:
    return identity_dir / "attempt"


def data_dir_in(run_dir: Path) -> Path:
    return run_dir / "data"


def scratch_dir_in(run_dir: Path) -> Path:
    return run_dir / "scratch"


def result_dir_in(run_dir: Path) -> Path:
    return run_dir / "result"


def result_manifest_path_in(run_dir: Path) -> Path:
    return result_dir_in(run_dir) / "manifest.json"


def provenance_path_in(run_dir: Path) -> Path:
    return run_dir / "provenance.json"


def trace_path_in(run_dir: Path) -> Path:
    return run_dir / "trace.json"


def trace_log_path_in(run_dir: Path) -> Path:
    return run_dir / "trace.log"


def run_log_path_in(identity_dir: Path) -> Path:
    return identity_dir / "run.log"


def schema_snapshot_path_in(identity_dir: Path) -> Path:
    # identity_dir is {fqn}/{schema_hash}/{artifact_hash}; the snapshot is
    # written once per (class, schema-hash), beside the identity directories.
    return schema_snapshot_path_in_schema_directory(identity_dir.parent)


def schema_snapshot_path_in_schema_directory(schema_directory: Path) -> Path:
    return schema_directory / "schema.json"


def execution_log_path_in(executor_dir: Path) -> Path:
    return executor_dir / "execution.log"


def slurm_worker_log_path_in(worker_dir: Path, worker: str) -> Path:
    # sbatch --output names the file after the job, and the worker is named
    # after the same job id, so either side can find it.
    return worker_dir / "logs" / f"{worker}.log"


def compute_lock_path_in(identity_dir: Path) -> Path:
    return identity_dir / "compute.lock"


def result_link_path_in(identity_dir: Path) -> Path:
    return identity_dir / "result-link.json"
