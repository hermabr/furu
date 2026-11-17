from pathlib import Path

# A class folder {storage}/{class name}/{class part} holds schema.ndjson and
# one identity directory per spec, named by its artifact hash. The class name
# is one path component, so the folders under it are exactly its class parts.
#
# An identity directory holds spec.json, compute.lock, run.log (every attempt's
# log), the in-progress attempt/ and one published version directory per code
# version (v-<hash>, v-fixed or v-fixed-<hash>). attempt/ and every version
# directory share the run-directory layout below.


def spec_path_in(identity_dir: Path) -> Path:
    return identity_dir / "spec.json"


def attempt_dir_in(identity_dir: Path) -> Path:
    return identity_dir / "attempt"


def version_dir_in(identity_dir: Path) -> Path:
    return identity_dir / "v-fixed"


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


def schema_path_in(class_dir: Path) -> Path:
    return class_dir / "schema.ndjson"


def schema_lock_path_in(class_dir: Path) -> Path:
    return class_dir / "schema.lock"


def execution_log_path_in(executor_dir: Path) -> Path:
    return executor_dir / "execution.log"


def compute_lock_path_in(identity_dir: Path) -> Path:
    return identity_dir / "compute.lock"
