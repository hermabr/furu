from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar, cast

from furu.results.api import ResultConfig
from furu.results.errors import ResultDeserializationError
from furu.results.nodes import BUNDLE_FORMAT, BUNDLE_VERSION
from furu.results.walker import dump_root, load_root
from furu.utils import JsonValue

T = TypeVar("T")


def read_strict_json(path: Path) -> JsonValue:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_strict_json(path: Path, value: JsonValue) -> None:
    text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.rename(path)


def write_manifest(path: Path, manifest: JsonValue) -> None:
    write_strict_json(path, manifest)


def dump_result_bundle(value: object, result_dir: Path, config: ResultConfig) -> None:
    result_dir.mkdir(parents=True, exist_ok=False)
    root = dump_root(value, bundle_dir=result_dir, config=config)
    manifest: JsonValue = {
        "format": BUNDLE_FORMAT,
        "root": root,
        "version": BUNDLE_VERSION,
    }
    write_manifest(result_dir / "manifest.json", manifest)


def load_result_bundle(result_dir: Path, config: ResultConfig) -> T:
    manifest = read_strict_json(result_dir / "manifest.json")
    if not isinstance(manifest, dict):
        raise ResultDeserializationError("Result manifest must be a JSON object")
    if (
        manifest.get("format") != BUNDLE_FORMAT
        or manifest.get("version") != BUNDLE_VERSION
    ):
        raise ResultDeserializationError(
            "Unsupported result manifest format or version"
        )
    if set(manifest) != {"format", "root", "version"}:
        raise ResultDeserializationError(
            "Result manifest has unexpected top-level keys"
        )
    return cast(T, load_root(manifest["root"], bundle_dir=result_dir, config=config))
