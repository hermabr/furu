import datetime
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ..config import GREN_CONFIG
from ..serialization import BaseModel as PydanticBaseModel
from ..serialization import GrenSerializer
from ..serialization.serializer import JsonValue

if TYPE_CHECKING:
    from ..core.gren import Gren


class GitInfo(BaseModel):
    """Git repository information."""

    model_config = ConfigDict(extra="forbid", strict=True)

    git_commit: str
    git_branch: str
    git_remote: str | None
    git_patch: str
    git_submodules: dict[str, str]


class EnvironmentInfo(BaseModel):
    """Runtime environment information."""

    model_config = ConfigDict(extra="forbid", strict=True)

    timestamp: str
    command: str
    python_version: str
    executable: str
    platform: str
    hostname: str
    user: str
    pid: int


class GrenMetadata(BaseModel):
    """Complete metadata for a Gren experiment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # Gren-specific fields
    gren_python_def: str
    gren_obj: JsonValue  # Serialized Gren object from GrenSerializer.to_dict()
    gren_hash: str
    gren_path: str

    # Git info
    git_commit: str
    git_branch: str
    git_remote: str | None
    git_patch: str
    git_submodules: dict[str, str]

    # Environment info
    timestamp: str
    command: str
    python_version: str
    executable: str
    platform: str
    hostname: str
    user: str
    pid: int


class MetadataManager:
    """Handles metadata collection and storage."""

    INTERNAL_DIR = ".gren"
    METADATA_FILE = "metadata.json"

    @classmethod
    def get_metadata_path(cls, directory: Path) -> Path:
        return directory / cls.INTERNAL_DIR / cls.METADATA_FILE

    @staticmethod
    def run_git_command(args: list[str]) -> str:
        """Run git command, return output."""
        proc = subprocess.run(
            ["git", *args], text=True, capture_output=True, timeout=10
        )
        if proc.returncode not in (0, 1):
            proc.check_returncode()
        return proc.stdout.strip()

    @classmethod
    def collect_git_info(cls, ignore_diff: bool = False) -> GitInfo:
        """Collect git repository information."""
        if not GREN_CONFIG.require_git:
            try:
                head = cls.run_git_command(["rev-parse", "HEAD"])
                branch = cls.run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
            except subprocess.CalledProcessError:
                return GitInfo(
                    git_commit="<no-git>",
                    git_branch="<no-git>",
                    git_remote=None,
                    git_patch="<no-git>",
                    git_submodules={},
                )
        else:
            head = cls.run_git_command(["rev-parse", "HEAD"])
            branch = cls.run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])

        if GREN_CONFIG.require_git_remote:
            remote = cls.run_git_command(["remote", "get-url", "origin"])
        else:
            try:
                remote = cls.run_git_command(["remote", "get-url", "origin"])
            except subprocess.CalledProcessError:
                remote = None

        if ignore_diff:
            patch = "<ignored-diff>"
        else:
            unstaged = cls.run_git_command(["diff"])
            staged = cls.run_git_command(["diff", "--cached"])
            untracked = cls.run_git_command(
                ["ls-files", "--others", "--exclude-standard"]
            ).splitlines()

            untracked_patches = "\n".join(
                cls.run_git_command(["diff", "--no-index", "/dev/null", f])
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
                    "Use ignore_diff=True or GREN_IGNORE_DIFF=1"
                )

        submodules: dict[str, str] = {}
        for line in cls.run_git_command(["submodule", "status"]).splitlines():
            parts = line.split()
            if len(parts) >= 2:
                submodules[parts[1]] = parts[0]

        return GitInfo(
            git_commit=head,
            git_branch=branch,
            git_remote=remote,
            git_patch=patch,
            git_submodules=submodules,
        )

    @staticmethod
    def collect_environment_info() -> EnvironmentInfo:
        """Collect environment information."""
        return EnvironmentInfo(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="microseconds"
            ),
            command=" ".join(sys.argv) if sys.argv else "<unknown>",
            python_version=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            hostname=socket.gethostname(),
            user=getpass.getuser(),
            pid=os.getpid(),
        )

    @classmethod
    def create_metadata(
        cls, gren_obj: "Gren", directory: Path, ignore_diff: bool = False
    ) -> GrenMetadata:
        """Create complete metadata for a Gren object."""
        git_info = cls.collect_git_info(ignore_diff)
        env_info = cls.collect_environment_info()

        serialized_obj = GrenSerializer.to_dict(gren_obj)
        if not isinstance(serialized_obj, dict):
            raise TypeError(
                f"Expected GrenSerializer.to_dict to return dict, got {type(serialized_obj)}"
            )

        return GrenMetadata(
            gren_python_def=GrenSerializer.to_python(gren_obj, multiline=False),
            gren_obj=serialized_obj,
            gren_hash=GrenSerializer.compute_hash(gren_obj),
            gren_path=str(directory.resolve()),
            git_commit=git_info.git_commit,
            git_branch=git_info.git_branch,
            git_remote=git_info.git_remote,
            git_patch=git_info.git_patch,
            git_submodules=git_info.git_submodules,
            timestamp=env_info.timestamp,
            command=env_info.command,
            python_version=env_info.python_version,
            executable=env_info.executable,
            platform=env_info.platform,
            hostname=env_info.hostname,
            user=env_info.user,
            pid=env_info.pid,
        )

    @classmethod
    def write_metadata(cls, metadata: GrenMetadata, directory: Path) -> None:
        """Write metadata to file."""
        metadata_path = cls.get_metadata_path(directory)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                metadata.model_dump(mode="json"),
                indent=2,
                default=lambda o: o.model_dump()
                if PydanticBaseModel is not None and isinstance(o, PydanticBaseModel)
                else str(o),
            )
        )

    @classmethod
    def read_metadata(cls, directory: Path) -> GrenMetadata:
        """Read metadata from file."""
        metadata_path = cls.get_metadata_path(directory)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        data = json.loads(metadata_path.read_text())
        return GrenMetadata.model_validate(data)

    @classmethod
    def read_metadata_raw(cls, directory: Path) -> dict[str, JsonValue] | None:
        """Read raw metadata JSON from file, returning None if not found."""
        metadata_path = cls.get_metadata_path(directory)
        if not metadata_path.is_file():
            return None
        return json.loads(metadata_path.read_text())
