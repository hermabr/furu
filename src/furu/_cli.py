from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ByteSize

from furu.provenance import Provenance
from furu.snapshot import extract_snapshot, read_snapshot_manifest
from furu.storage._layout import metadata_path_in, provenance_path_in


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="furu", description="furu command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    repro = subparsers.add_parser(
        "repro",
        help="re-materialize the code and environment that produced a result",
    )
    repro.add_argument(
        "result",
        type=Path,
        help="path to a stored result directory (the one holding provenance.json)",
    )
    repro.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="directory to extract the snapshot into "
        "(default: ./repro-<snapshot prefix>)",
    )
    repro.add_argument(
        "--verify",
        action="store_true",
        help="re-hash the extracted tree and check it against the snapshot id",
    )
    repro.add_argument(
        "--no-sync",
        action="store_true",
        help="skip running `uv sync --frozen` in the extracted directory",
    )
    args = parser.parse_args(argv)
    try:
        return _repro(
            args.result, dest=args.dest, verify=args.verify, sync=not args.no_sync
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _repro(result: Path, *, dest: Path | None, verify: bool, sync: bool) -> int:
    result_dir, provenance = _load_provenance(result)
    _print_summary(result_dir, provenance)

    if provenance.snapshot_id is None:
        _print_no_snapshot_instructions(provenance)
        return 0

    if dest is None:
        dest = Path(f"repro-{provenance.snapshot_id[:6]}")
    extract_snapshot(provenance.snapshot_id, dest, verify=verify)
    print(
        f"✓ extracted snapshot → {dest}"
        + ("  (manifest hash verified)" if verify else "")
    )
    if sync:
        _uv_sync(dest)

    print()
    print("To re-run:")
    print(f"  cd {dest}")
    print(f"  {shlex.join(provenance.submitted.launch_command)}")
    return 0


def _load_provenance(result: Path) -> tuple[Path, Provenance]:
    start = result if result.is_dir() else result.parent
    for candidate in (start, *start.parents):
        path = provenance_path_in(candidate)
        if path.is_file():
            return candidate, Provenance.model_validate_json(
                path.read_text(encoding="utf-8")
            )
    raise RuntimeError(
        f"no provenance.json found at or above {result}.\n"
        "Pass a stored result directory (the one holding metadata.json and "
        "provenance.json), e.g. furu-data/objects/<spec>/<schema>/<artifact>."
    )


def _print_summary(result_dir: Path, provenance: Provenance) -> None:
    if (spec := _spec_line(result_dir)) is not None:
        print(f"{'spec':<12}{spec}")

    git = provenance.git
    commit = git.commit[:7]
    if git.branch is not None:
        commit += f" · {git.branch}"
    if git.dirty:
        commit += " · dirty" + (f" ({git.diff_stats})" if git.diff_stats else "")
    print(f"{'commit':<12}{commit}")

    if provenance.snapshot_id is None:
        print(f"{'snapshot':<12}none (snapshotting was off)")
    else:
        manifest = read_snapshot_manifest(provenance.snapshot_id)
        size = ByteSize(manifest.total_bytes).human_readable(separator=" ")
        print(
            f"{'snapshot':<12}{provenance.snapshot_id} · {size} · "
            f"{len(manifest.entries):,} files"
        )

    executed = provenance.executed
    parts = [executed.hostname]
    if executed.accelerators:
        parts.append(", ".join(executed.accelerators))
    if executed.slurm_job_id is not None:
        parts.append(f"slurm {executed.slurm_job_id}")
    parts.append(provenance.submitted.timestamp.date().isoformat())
    print(f"{'executed':<12}{' · '.join(parts)}")
    print()


def _spec_line(result_dir: Path) -> str | None:
    try:
        artifact = json.loads(metadata_path_in(result_dir).read_text(encoding="utf-8"))[
            "artifact"
        ]
        data = artifact["artifact_data"]
        if isinstance(data.get("|fields"), dict):  # unwrap the instance envelope
            data = data["|fields"]
        arguments = ", ".join(
            f"{name}={json.dumps(value)}" for name, value in data.items()
        )
        return f"{artifact['fully_qualified_name']}({arguments})"
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
        return None


def _uv_sync(dest: Path) -> None:
    print(f"running uv sync --frozen in {dest} …")
    try:
        completed = subprocess.run(["uv", "sync", "--frozen"], cwd=dest)
    except OSError as exc:
        raise RuntimeError(
            "uv executable not found\n"
            "Install uv (https://docs.astral.sh/uv/) and re-run, or pass "
            "--no-sync to skip environment rebuilding."
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"uv sync --frozen failed with exit code {completed.returncode} in {dest}"
        )
    print("✓ uv sync --frozen")


def _print_no_snapshot_instructions(provenance: Provenance) -> None:
    git = provenance.git
    print("This result was computed without a snapshot; to reproduce it manually:")
    if git.remote is not None:
        print(f"  git clone {git.remote}")
    print(f"  git checkout {git.commit}")
    print("  uv sync --frozen")
    print(f"  {shlex.join(provenance.submitted.launch_command)}")
    if git.dirty:
        print()
        print(
            "warning: the submitting worktree was dirty"
            + (f" ({git.diff_stats})" if git.diff_stats else "")
            + "; those changes were not captured, so this is only "
            "approximately reproducible."
        )


if __name__ == "__main__":
    sys.exit(main())
