import datetime
import getpass
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

from ..config import FURU_CONFIG
from ..schema import schema_key_from_furu_obj
from ..serialization import BaseModel as PydanticBaseModel
from ..serialization import FuruSerializer
from ..serialization.serializer import JsonValue

if TYPE_CHECKING:
    from ..core.furu import Furu

# Module-level cache for metadata (controlled via FURU_RECORD_GIT=cached)
_cached_git_info: "GitInfo | None" = None
_cached_git_info_time: float = 0.0


def clear_metadata_cache() -> None:
    """Clear the cached metadata. Useful for testing or long-running processes."""
    global _cached_git_info, _cached_git_info_time
    _cached_git_info = None
    _cached_git_info_time = 0.0


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
    furu_version: str | None = None
    python_version: str
    executable: str
    platform: str
    hostname: str
    user: str
    pid: int


class FuruMetadata(BaseModel):
    """Complete metadata for a Furu experiment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # Furu-specific fields
    furu_python_def: str
    furu_obj: JsonValue  # Serialized Furu object from FuruSerializer.to_dict()
    schema_key: tuple[str, ...]
    furu_hash: str
    furu_path: str

    # Git info
    git_commit: str
    git_branch: str
    git_remote: str | None
    git_patch: str
    git_submodules: dict[str, str]

    # Environment info
    timestamp: str
    command: str
    furu_version: str | None = None
    python_version: str
    executable: str
    platform: str
    hostname: str
    user: str
    pid: int

    @field_validator("schema_key", mode="before")
    @classmethod
    def _coerce_schema_key(cls, value: object) -> object:
        """
        Coerce a provided schema_key into a tuple when it's given as a list.
        
        Parameters:
            cls: The class on which the validator runs.
            value (object): The incoming value for `schema_key`; may be a list or already a tuple.
        
        Returns:
            object: A tuple converted from `value` if `value` is a list, otherwise the original `value`.
        """
        if isinstance(value, list):
            return tuple(value)
        return value


class MetadataManager:
    """Handles metadata collection and storage."""

    INTERNAL_DIR = ".furu"
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
        global _cached_git_info, _cached_git_info_time
        import time

        record_git = FURU_CONFIG.record_git
        if record_git == "ignore":
            return GitInfo(
                git_commit="<ignored>",
                git_branch="<ignored>",
                git_remote=None,
                git_patch="<ignored>",
                git_submodules={},
            )

        ttl = FURU_CONFIG.cache_metadata_ttl_sec
        # Return cached result if caching is enabled and not expired
        if ttl is not None and _cached_git_info is not None:
            age = time.time() - _cached_git_info_time
            if age < ttl:
                return _cached_git_info

        try:
            head = cls.run_git_command(["rev-parse", "HEAD"])
            branch = cls.run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Failed to read git commit/branch for provenance. "
                "If this is expected, set FURU_RECORD_GIT=ignore."
            ) from e

        if FURU_CONFIG.allow_no_git_origin:
            try:
                remote = cls.run_git_command(["remote", "get-url", "origin"])
            except (subprocess.CalledProcessError, FileNotFoundError):
                remote = None
        else:
            try:
                remote = cls.run_git_command(["remote", "get-url", "origin"])
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError(
                    "Git remote 'origin' is required for provenance but was not found. "
                    "Set FURU_ALLOW_NO_GIT_ORIGIN=1 to allow missing origin, "
                    "or set FURU_RECORD_GIT=ignore to disable git metadata."
                ) from e

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
                    "Set FURU_RECORD_GIT=ignore to skip git metadata."
                )

        submodules: dict[str, str] = {}
        for line in cls.run_git_command(["submodule", "status"]).splitlines():
            parts = line.split()
            if len(parts) >= 2:
                submodules[parts[1]] = parts[0]

        result = GitInfo(
            git_commit=head,
            git_branch=branch,
            git_remote=remote,
            git_patch=patch,
            git_submodules=submodules,
        )

        # Cache result if caching is enabled
        if ttl is not None:
            _cached_git_info = result
            _cached_git_info_time = time.time()

        return result

    @staticmethod
    def collect_environment_info() -> EnvironmentInfo:
        """Collect environment information."""
        return EnvironmentInfo(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="microseconds"
            ),
            command=" ".join(sys.argv) if sys.argv else "<unknown>",
            furu_version=importlib.metadata.version("furu"),
            python_version=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            hostname=socket.gethostname(),
            user=getpass.getuser(),
            pid=os.getpid(),
        )

    @classmethod
    def create_metadata(
        cls, furu_obj: "Furu", directory: Path, ignore_diff: bool = False
    ) -> FuruMetadata:
        """
        Create a FuruMetadata object containing serialized Furu data, git provenance, and environment information.
        
        Parameters:
            furu_obj (Furu): The Furu object to serialize and include in the metadata.
            directory (Path): Filesystem directory containing the Furu object; stored as the resolved path.
            ignore_diff (bool): If True, omit detailed git diff contents from the metadata.
        
        Returns:
            FuruMetadata: A validated metadata model with fields for the serialized Furu object, a schema key derived from that serialized object, a content hash, filesystem path, git commit/branch/remote/patch/submodules, and environment details (timestamp, command, versions, executable, platform, hostname, user, pid).
        
        Raises:
            TypeError: If serialization of `furu_obj` does not produce a dict.
        """
        git_info = cls.collect_git_info(ignore_diff)
        env_info = cls.collect_environment_info()

        serialized_obj = FuruSerializer.to_dict(furu_obj)
        if not isinstance(serialized_obj, dict):
            raise TypeError(
                f"Expected FuruSerializer.to_dict to return dict, got {type(serialized_obj)}"
            )
        schema_key = schema_key_from_furu_obj(serialized_obj)

        return FuruMetadata(
            furu_python_def=FuruSerializer.to_python(furu_obj, multiline=False),
            furu_obj=serialized_obj,
            schema_key=schema_key,
            furu_hash=FuruSerializer.compute_hash(furu_obj),
            furu_path=str(directory.resolve()),
            git_commit=git_info.git_commit,
            git_branch=git_info.git_branch,
            git_remote=git_info.git_remote,
            git_patch=git_info.git_patch,
            git_submodules=git_info.git_submodules,
            timestamp=env_info.timestamp,
            command=env_info.command,
            furu_version=env_info.furu_version,
            python_version=env_info.python_version,
            executable=env_info.executable,
            platform=env_info.platform,
            hostname=env_info.hostname,
            user=env_info.user,
            pid=env_info.pid,
        )

    @classmethod
    def write_metadata(cls, metadata: FuruMetadata, directory: Path) -> None:
        """Write metadata to file."""
        metadata_path = cls.get_metadata_path(directory)
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
    def read_metadata(cls, directory: Path) -> FuruMetadata:
        """Read metadata from file."""
        metadata_path = cls.get_metadata_path(directory)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        data = json.loads(metadata_path.read_text())
        return FuruMetadata.model_validate(data)

    @classmethod
    def read_metadata_raw(cls, directory: Path) -> dict[str, JsonValue] | None:
        """Read raw metadata JSON from file, returning None if not found."""
        metadata_path = cls.get_metadata_path(directory)
        if not metadata_path.is_file():
            return None
        return json.loads(metadata_path.read_text())