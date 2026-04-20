from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

from furu.config import ResultConfig
from furu.locking import LockLostError
from furu.results.codecs import _write_json
from furu.results.nodes import ManifestValue, RESULT_FORMAT, make_result_manifest
from furu.results.walker import load_manifest_root, dump_manifest_root
from furu.utils import _nfs_safe_unique_name

type HasLock = Callable[[], bool]


def save_result_bundle[T](
    value: T,
    result_dir: Path,
    config: ResultConfig,
    *,
    has_lock: HasLock | None = None,
    lock_path: Path | None = None,
) -> None:
    has_lock = has_lock or (lambda: True)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = _nfs_safe_unique_name(result_dir, name="tmp")

    try:
        root = dump_manifest_root(value, result_dir=staging_dir, config=config)
        staging_dir.mkdir(parents=True, exist_ok=True)
        _write_json(staging_dir / "manifest.json", make_result_manifest(root))

        if not has_lock():
            raise LockLostError(_lock_lost_message(lock_path))

        staging_dir.rename(result_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def load_result_bundle[T](result_dir: Path, config: ResultConfig) -> T:
    raw = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format") != RESULT_FORMAT:
        raise ValueError(
            f"Unsupported result manifest at {result_dir / 'manifest.json'}"
        )
    return cast(
        T,
        load_manifest_root(
            cast(ManifestValue, raw["root"]), result_dir=result_dir, config=config
        ),
    )


def _lock_lost_message(lock_path: Path | None) -> str:
    if lock_path is None:
        return "lost lock before writing final result"
    return f"lost lock at {lock_path} before writing final result"
