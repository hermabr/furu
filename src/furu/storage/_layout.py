import os
from collections.abc import Sequence
from pathlib import Path

from furu.utils import _hash_dict_deterministically

# A class folder {storage}/{class name}/{class part} holds schema.ndjson,
# attempts/, one folder per code version, and remembered/ when the class has a
# @furu.dependency(remember=True) method. The class name is one path component,
# so the folders under it are exactly its class parts.
#
#   attempts/<run>/<owner>    the working folder of a run in progress; its name
#                             encodes ownership (free__, slurm__, or local__)
#   <code version>/trace.json which code its runs ran; "fixed" has none
#   <code version>/runs/<run> a published run, which never changes again
#   remembered/               what such methods returned, a file per launch
#
# A working folder is renamed into runs/ under the logical run name, so both
# share the run layout below. A run holds one row per spec: the specs of one
# batch, or the one spec of a class without @furu.batched.

FIXED = "fixed"
REMEMBERED = "remembered"
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


def attempt_parent_in(class_dir: Path, run: str) -> Path:
    return attempts_dir_in(class_dir) / run


def version_dirs_in(class_dir: Path) -> list[Path]:
    """The class's code-version folders: not attempts/, remembered/ or the schema files."""
    try:
        names = sorted(os.listdir(class_dir))
    except FileNotFoundError:
        return []
    return [
        class_dir / name
        for name in names
        if name not in ("attempts", REMEMBERED) and "." not in name
    ]


def remembered_dir_in(class_dir: Path) -> Path:
    return class_dir / REMEMBERED


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


def trace_log_path_in(run_dir: Path) -> Path:
    return run_dir / "trace.log"


def schema_path_in(class_dir: Path) -> Path:
    return class_dir / "schema.ndjson"


def schema_lock_path_in(class_dir: Path) -> Path:
    return class_dir / "schema.lock"


def execution_log_path_in(executor_dir: Path) -> Path:
    return executor_dir / "execution.log"
