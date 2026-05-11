from pathlib import Path


def _result_dir_in(data_dir: Path) -> Path:
    return data_dir / "result"


def _result_manifest_path_in(data_dir: Path) -> Path:
    return _result_dir_in(data_dir) / "manifest.json"


def _internal_furu_dir_in(data_dir: Path) -> Path:
    return data_dir / ".furu"


def _metadata_path_in(data_dir: Path) -> Path:
    return _internal_furu_dir_in(data_dir) / "metadata.json"


def _result_link_path_in(data_dir: Path) -> Path:
    return _internal_furu_dir_in(data_dir) / "result-link.json"
