import os
from collections.abc import Sequence
from pathlib import Path

from furu.utils import _hash_dict_deterministically

# A class folder {storage}/{class name}/{class part} holds schema.ndjson,
# attempts/ and one folder per code version. The class name is one path
# component, so the folders under it are exactly its class parts.
#
#   attempts/<run>/           the working folder of a run in progress
#   attempts/<run>.lock       its lock, beside it so that publishing never moves it
#   <code version>/trace.json which code its runs ran; "fixed" has none
#   <code version>/runs/<run> a published run, which never changes again
#
# A working folder is renamed into runs/ under the same name, so both share
# the run layout below. A run holds one row per spec: the specs of one batch,
# or the one spec of a class without @furu.batched.

FIXED = "fixed"
# Starts the name of a run with more than one row; a spec hash is hex only.
BATCH = "batch-"


def run_name(spec_hashes: Sequence[str]) -> str:
    """A run is named by the hashes of its specs in row order.

    One spec's run has that spec's own hash, so it is found by name. A batch's
    name starts with BATCH, so a listing tells the two apart.
    """
    if len(spec_hashes) == 1:
        return spec_hashes[0]
    return BATCH + _hash_dict_deterministically(list(spec_hashes))


def attempts_dir_in(class_dir: Path) -> Path:
    return class_dir / "attempts"


def attempt_dir_in(class_dir: Path, run: str) -> Path:
    return attempts_dir_in(class_dir) / run


def lock_path_in(class_dir: Path, run: str) -> Path:
    return attempts_dir_in(class_dir) / f"{run}.lock"


def version_dirs_in(class_dir: Path) -> list[Path]:
    """The class's code-version folders: all but attempts/ and the schema files."""
    try:
        names = sorted(os.listdir(class_dir))
    except FileNotFoundError:
        return []
    return [
        class_dir / name for name in names if name != "attempts" and "." not in name
    ]


def trace_path_in(version_dir: Path) -> Path:
    return version_dir / "trace.json"


def runs_dir_in(version_dir: Path) -> Path:
    return version_dir / "runs"


def run_dir_in(version_dir: Path, run: str) -> Path:
    return runs_dir_in(version_dir) / run


def events_path_in(run_dir: Path) -> Path:
    return run_dir / "events.jsonl"


def log_path_in(run_dir: Path) -> Path:
    return run_dir / "log"


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


def trace_log_path_in(run_dir: Path) -> Path:
    return run_dir / "trace.log"


def schema_path_in(class_dir: Path) -> Path:
    return class_dir / "schema.ndjson"


def schema_lock_path_in(class_dir: Path) -> Path:
    return class_dir / "schema.lock"


def execution_log_path_in(executor_dir: Path) -> Path:
    return executor_dir / "execution.log"
