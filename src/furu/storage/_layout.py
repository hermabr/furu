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


def provenance_path_in(run_dir: Path) -> Path:
    return run_dir / "provenance.json"


def trace_path_in(run_dir: Path) -> Path:
    return run_dir / "trace.json"


def trace_log_path_in(run_dir: Path) -> Path:
    return run_dir / "trace.log"


def run_log_path_in(identity_dir: Path) -> Path:
    return identity_dir / "run.log"


def schema_snapshot_path_in(schema_dir: Path) -> Path:
    # schema_dir is {fqn}/{schema_hash}: the snapshot is written once per
    # (class, schema-hash), beside the identity directories.
    return schema_dir / "schema.json"


def migrations_path_in(schema_dir: Path) -> Path:
    # The migration steps declared by the code that first wrote this schema;
    # written once, just before the schema snapshot.
    return schema_dir / "migrations.json"


def execution_log_path_in(executor_dir: Path) -> Path:
    return executor_dir / "execution.log"


def compute_lock_path_in(identity_dir: Path) -> Path:
    return identity_dir / "compute.lock"


def result_link_path_in(identity_dir: Path) -> Path:
    return identity_dir / "result-link.json"
