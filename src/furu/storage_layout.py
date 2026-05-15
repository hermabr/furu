from pathlib import Path


def data_dir_in(base_dir: Path) -> Path:
    return base_dir / "data"


def result_dir_in(base_dir: Path) -> Path:
    return base_dir / "result"


def result_manifest_path_in(base_dir: Path) -> Path:
    return result_dir_in(base_dir) / "manifest.json"


def ensure_object_dirs_in(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    data_dir_in(base_dir).mkdir(parents=True, exist_ok=True)


def metadata_path_in(base_dir: Path) -> Path:
    return base_dir / "metadata.json"


def run_log_path_in(base_dir: Path) -> Path:
    return base_dir / "run.log"


def manager_log_path_in(executor_dir: Path) -> Path:
    return executor_dir / "manager.log"


def compute_lock_path_in(base_dir: Path) -> Path:
    return base_dir / "compute.lock"


def result_link_path_in(base_dir: Path) -> Path:
    return base_dir / "result-link.json"
