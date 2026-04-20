from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import ResultConfig
from .errors import ResultLoadError
from .paths import LogicalPath
from .walker import dump_root, load_root


def save_result_bundle(value: Any, result_dir: Path, config: ResultConfig) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "artifacts").mkdir(exist_ok=True)
    manifest = {
        "format": "furu.result",
        "version": 1,
        "root": dump_root(value, bundle_dir=result_dir, config=config),
    }
    manifest_path = result_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def load_result_bundle(result_dir: Path, config: ResultConfig) -> Any:
    logical_path = LogicalPath()
    manifest_path = result_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ResultLoadError(
            logical_path=logical_path,
            detail="manifest root must be a JSON object.",
        )
    if manifest.get("format") != "furu.result":
        raise ResultLoadError(
            logical_path=logical_path,
            detail="unsupported manifest format.",
        )
    if manifest.get("version") != 1:
        raise ResultLoadError(
            logical_path=logical_path,
            detail="unsupported manifest version.",
        )
    if "root" not in manifest:
        raise ResultLoadError(
            logical_path=logical_path, detail="manifest is missing root."
        )
    return load_root(manifest["root"], bundle_dir=result_dir, config=config)
