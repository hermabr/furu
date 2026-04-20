from __future__ import annotations

import json
import os
from pathlib import Path


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2))


def _fsync_path(path: Path) -> None:
    with path.open("rb") as f:
        os.fsync(f.fileno())
