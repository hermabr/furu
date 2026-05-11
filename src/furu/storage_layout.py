from pathlib import Path


RESULT_DIR_NAME = "result"
RESULT_MANIFEST_FILE_NAME = "manifest.json"
INTERNAL_FURU_DIR_NAME = ".furu"
METADATA_FILE_NAME = "metadata.json"
RUN_LOG_FILE_NAME = "run.log"
COMPUTE_LOCK_FILE_NAME = "compute.lock"
RESULT_LINK_FILE_NAME = "result-link.json"


def result_dir_in(data_dir: Path) -> Path:
    return data_dir / RESULT_DIR_NAME


def result_manifest_path_in(data_dir: Path) -> Path:
    return result_dir_in(data_dir) / RESULT_MANIFEST_FILE_NAME


def internal_furu_dir_in(data_dir: Path) -> Path:
    return data_dir / INTERNAL_FURU_DIR_NAME


def metadata_path_in(data_dir: Path) -> Path:
    return internal_furu_dir_in(data_dir) / METADATA_FILE_NAME


def run_log_path_in(data_dir: Path) -> Path:
    return internal_furu_dir_in(data_dir) / RUN_LOG_FILE_NAME


def compute_lock_path_in(data_dir: Path) -> Path:
    return internal_furu_dir_in(data_dir) / COMPUTE_LOCK_FILE_NAME


def result_link_path_in(data_dir: Path) -> Path:
    return internal_furu_dir_in(data_dir) / RESULT_LINK_FILE_NAME
