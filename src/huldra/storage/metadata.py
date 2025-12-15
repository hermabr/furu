import datetime
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..serialization import BaseModel, HuldraSerializer


class MetadataManager:
    """Handles metadata collection and storage."""

    @staticmethod
    def safe_git_command(args: List[str]) -> str:
        """Run git command safely, return output or error message."""
        try:
            proc = subprocess.run(
                ["git", *args], text=True, capture_output=True, timeout=10
            )
            if proc.returncode not in (0, 1):
                proc.check_returncode()
            return proc.stdout.strip()
        except Exception as exc:
            print(f"WARNING: git {' '.join(args)} failed: {exc}", file=sys.stderr)
            return "<unavailable>"

    @classmethod
    def collect_git_info(cls, ignore_diff: bool = False) -> Dict[str, Any]:
        """Collect git repository information."""
        head = cls.safe_git_command(["rev-parse", "HEAD"])
        branch = cls.safe_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        remote = cls.safe_git_command(["remote", "get-url", "origin"])

        if ignore_diff:
            patch = "<ignored-diff>"
        else:
            unstaged = cls.safe_git_command(["diff"])
            staged = cls.safe_git_command(["diff", "--cached"])
            untracked = cls.safe_git_command(
                ["ls-files", "--others", "--exclude-standard"]
            ).splitlines()

            untracked_patches = "\n".join(
                cls.safe_git_command(["diff", "--no-index", "/dev/null", f])
                for f in untracked
            )

            patch = "\n".join(
                filter(
                    None,
                    [
                        "# === unstaged ==================================================",
                        unstaged,
                        "# === staged ====================================================",
                        staged,
                        "# === untracked ================================================",
                        untracked_patches,
                    ],
                )
            )

            if len(patch) > 50_000:
                raise ValueError(
                    f"Git diff too large ({len(patch):,} bytes). "
                    "Use ignore_diff=True or HULDRA_IGNORE_DIFF=1"
                )

        submodules = {}
        for line in cls.safe_git_command(["submodule", "status"]).splitlines():
            parts = line.split()
            if len(parts) >= 2:
                submodules[parts[1]] = parts[0]

        return {
            "git_commit": head,
            "git_branch": branch,
            "git_remote": remote,
            "git_patch": patch,
            "git_submodules": submodules,
        }

    @staticmethod
    def collect_environment_info() -> Dict[str, Any]:
        """Collect environment information."""
        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "command": " ".join(sys.argv) if sys.argv else "<unknown>",
            "python_version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
        }

    @classmethod
    def create_metadata(
        cls, huldra_obj: Any, directory: Path, ignore_diff: bool = False
    ) -> Dict[str, Any]:
        """Create complete metadata dictionary."""
        metadata = {
            "huldra_python_def": HuldraSerializer.to_python(
                huldra_obj, multiline=False
            ),
            "huldra_obj": HuldraSerializer.to_dict(huldra_obj),
            "huldra_hash": HuldraSerializer.compute_hash(huldra_obj),
            "huldra_path": str(directory.resolve()),
            **cls.collect_git_info(ignore_diff),
            **cls.collect_environment_info(),
        }
        return metadata

    @classmethod
    def write_metadata(cls, metadata: Dict[str, Any], directory: Path) -> None:
        """Write metadata to file."""
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                default=lambda o: o.model_dump()
                if BaseModel is not None and isinstance(o, BaseModel)
                else str(o),
            )
        )

    @classmethod
    def read_metadata(cls, directory: Path) -> Dict[str, Any]:
        """Read metadata from file."""
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        return json.loads(metadata_path.read_text())
