from pathlib import Path


def result_dir_in(data_dir: Path) -> Path:
    return data_dir / "result"


def result_manifest_path_in(data_dir: Path) -> Path:
    return result_dir_in(data_dir) / "manifest.json"


def internal_furu_dir_in(data_dir: Path) -> Path:
    return data_dir / ".furu"


def metadata_path_in(data_dir: Path) -> Path:
    return internal_furu_dir_in(data_dir) / "metadata.json"


def run_log_path_in(data_dir: Path) -> Path:
    return internal_furu_dir_in(data_dir) / "run.log"


def compute_lock_path_in(data_dir: Path) -> Path:
    return internal_furu_dir_in(data_dir) / "compute.lock"


def result_link_path_in(data_dir: Path) -> Path:
    return internal_furu_dir_in(data_dir) / "result-link.json"
