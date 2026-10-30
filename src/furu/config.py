import functools
import json
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, ByteSize, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)

_WORKER_JSON_CONFIG_FILE_ENV_VAR = "_FURU_WORKER_JSON_CONFIG_FILE"


def _reject_submodules(repo_root: Path) -> None:
    """Crash if ``repo_root`` is or contains a git submodule.

    Provenance and code snapshots only see one repository's files, so code in
    a submodule would silently go unrecorded.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()

    superproject = git("rev-parse", "--show-superproject-working-tree")
    if superproject:
        raise RuntimeError(
            f"furu does not support git submodules, but {repo_root} is a "
            f"submodule of {superproject}.\n"
            "Are you sure you put your project in the correct place?"
        )
    submodules = [
        line.partition("\t")[2]
        for line in git("ls-files", "-s", "-z").split("\0")
        if line.startswith("160000 ")
    ]
    if submodules:
        raise RuntimeError(
            f"furu does not support git submodules, but {repo_root} contains:\n"
            + "".join(f"  {path}\n" for path in submodules)
            + "Are you sure you put them in the correct place?"
        )


@functools.cache
def _project_anchor() -> Path:
    """The main worktree's root, so all worktrees share one set of data directories.

    furu requires a git repository without submodules and with a pyproject.toml
    inside it. The pyproject.toml never moves the anchor; it is required because
    provenance records the nearest one (with its uv.lock), and code snapshots
    only carry files under the git root.
    """
    cwd = Path.cwd().resolve()
    try:
        toplevel, common_dir = subprocess.run(
            [
                "git",
                "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"no git repository found from {cwd} upward.\n"
            "furu anchors its data directories at the git root. Create one with:\n"
            "  git init"
        ) from exc
    _reject_submodules(Path(toplevel))
    within_repo = (cwd, *cwd.parents)[: len(cwd.relative_to(toplevel).parts) + 1]
    if not any((d / "pyproject.toml").is_file() for d in within_repo):
        raise RuntimeError(
            f"no pyproject.toml found from {cwd} up to the git root {toplevel}.\n"
            "furu requires a uv project inside the git repository so results are "
            "reproducible. Create one with:\n"
            "  uv init"
        )
    return Path(common_dir).parent


class _FuruDirectories(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objects: Path = Path("furu-data") / "objects"
    executions: Path = Path("furu-data") / "executions"
    snapshots: Path = Path("furu-data") / "snapshots"
    debug: Path = Path("furu-data") / "debug"

    def anchored(self) -> "_FuruDirectories":
        return _FuruDirectories(
            **{
                name: path if path.is_absolute() else _project_anchor() / path
                for name, path in self
            }
        )


class _FuruWorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connect_host: str | None = None
    idle_timeout_seconds: float = 60.0
    max_retries_per_object: int = 10
    max_failures_per_worker: int = 10


class _FuruProvenanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: bool = True
    max_snapshot_bytes: ByteSize = ByteSize(256 * 1024 * 1024)


class _Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FURU_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        pyproject_toml_depth=4,
        pyproject_toml_table_header=("tool", "furu"),
        extra="ignore",
        frozen=True,
    )

    debug_mode: bool = False
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories)
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)
    provenance: _FuruProvenanceConfig = Field(default_factory=_FuruProvenanceConfig)

    @property
    def run_directories(self) -> _FuruDirectories:
        directories = self.directories
        if self.debug_mode:
            debug = directories.debug
            directories = _FuruDirectories(
                **{name: debug / name for name in _FuruDirectories.model_fields}
                | {"debug": debug}
            )
        return directories.anchored()

    def anchored(self) -> "_Config":
        """This config with absolute directories, for processes whose working
        directory (e.g. a snapshot) must not decide where data lives."""
        return self.model_copy(update={"directories": self.directories.anchored()})

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            JsonConfigSettingsSource(
                settings_cls,
                json_file=os.environ.get(_WORKER_JSON_CONFIG_FILE_ENV_VAR),
                json_file_encoding="utf-8",
            ),
            env_settings,
            dotenv_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


_config = _Config()


def get_config() -> _Config:
    return _config


def _set_config(config: _Config) -> None:
    global _config
    _config = config


def _dump_worker_json_config(config: _Config, *, coordinator_url: str) -> str:
    document = config.model_dump(mode="json")
    document["coordinator_url"] = coordinator_url
    return json.dumps(document, indent=2) + "\n"


def _read_worker_json_config(path: Path) -> tuple[str, _Config]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    return document.pop("coordinator_url"), _Config.model_validate(document)
