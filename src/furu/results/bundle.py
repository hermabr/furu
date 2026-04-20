from __future__ import annotations

import json
from pathlib import Path

from furu.results.api import ResultConfig
from furu.results.nodes import manifest_document, manifest_root_from_document
from furu.results.walker import DumpContext, LoadContext, dump_manifest, load_manifest


def save_result_bundle(
    value: object,
    bundle_dir: Path,
    config: ResultConfig,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=False)
    root = dump_manifest(
        value,
        DumpContext(
            bundle_dir=bundle_dir,
            artifacts_dir=bundle_dir / "artifacts",
            logical_path=(),
            registry=config.registry,
            rules=tuple(config.rules),
        ),
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_document(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_result_bundle(
    bundle_dir: Path,
    config: ResultConfig,
) -> object:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing result manifest at {manifest_path}")
    root = manifest_root_from_document(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    return load_manifest(
        root,
        LoadContext(
            bundle_dir=bundle_dir,
            artifacts_dir=bundle_dir / "artifacts",
            logical_path=(),
            registry=config.registry,
        ),
    )
