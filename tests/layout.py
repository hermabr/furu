from pathlib import Path

from furu.storage._layout import result_dir_in


def version_dir_in(identity_dir: Path) -> Path:
    return identity_dir / "v-fixed"


def result_manifest_path_in(run_dir: Path) -> Path:
    return result_dir_in(run_dir) / "manifest.json"
