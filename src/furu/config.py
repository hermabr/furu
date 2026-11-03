import functools
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)


@functools.cache
def _project_anchor() -> Path:
    """The main worktree's root, so all worktrees share one set of data directories."""
    cwd = Path.cwd().resolve()
    try:
        toplevel, common_dir, *superproject = subprocess.run(
            [
                "git",
                "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
                "--git-common-dir",
                "--show-superproject-working-tree",
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
    if superproject or (Path(toplevel) / ".gitmodules").exists():
        raise RuntimeError(
            f"{toplevel} is or contains a git submodule, which furu does not "
            "support.\nAre you sure you put it in the correct place?"
        )
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

    idle_timeout_seconds: float = 60.0
    max_retries_per_object: int = 10


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
    # Where the furu manager publishes address.json; scripts on compute nodes
    # read it, so point it at storage they can see.
    manager_dir: Path = Path.home() / ".furu" / "manager"
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories)
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)

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
