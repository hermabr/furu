import functools
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)

# "fixed": a result is valid until you bump a field. "traced": furu derives the
# code version from the repo code create() ran (see furu.code_trace).
type CodeVersion = Literal["fixed", "traced"]


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
    # One place each per project, apart from the results, which each class
    # stores where its Metadata says. Every machine that runs a worker must
    # see them.
    plans: Path = Path("furu-data") / "plans"
    provenance: Path = Path("furu-data") / "provenance"
    snapshots: Path = Path("furu-data") / "snapshots"

    def anchored(self) -> "_FuruDirectories":
        return _FuruDirectories(
            **{
                name: path if path.is_absolute() else _project_anchor() / path
                for name, path in self
            }
        )


class _FuruWorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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

    # Each furu version's manager publishes address.json in its own
    # subdirectory; scripts on compute nodes read it, so point it at storage
    # they can see.
    manager_dir: Path = Path.home() / ".furu" / "manager"
    code_version: CodeVersion = "fixed"
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories)
    # A debug session. Whatever furu would write at an absolute path it writes
    # in this folder instead, at that path with its root taken off: results,
    # plans, records, snapshots and the manager's files alike. What is at the
    # path itself is still read, and never changed or locked, so deleting
    # this folder undoes the session. Every machine that runs a worker must
    # see it. Nothing else differs: a launch still takes committed, pushed
    # code.
    overlay: Path | None = None
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)

    @field_validator("overlay", mode="before")
    @classmethod
    def _checked_overlay(cls, overlay: object) -> object:
        if overlay == "":
            return None  # FURU_OVERLAY= is no overlay
        if not isinstance(overlay, str | Path):
            return overlay
        # A path is in the overlay when it starts with it, as written here.
        path = Path(os.path.normpath(overlay))
        if not path.is_absolute() or path == Path(path.anchor):
            raise ValueError(
                f"overlay must be an absolute path to a folder of its own, got "
                f"{overlay}. A relative one is not anchored at the project as "
                "the directories are."
            )
        return path

    @property
    def manager_version_dir(self) -> Path:
        """This furu version's manager: one per version, so scripts never meet another.

        An overlay has its own, as a manager knows a spec by its id alone: one
        that served a debug session too would take its specs for the stored ones.
        """
        return self.overlaid(self.manager_dir) / version("furu")

    def anchored(self) -> "_Config":
        """This config with absolute directories, for processes whose working
        directory (e.g. a snapshot) must not decide where data lives."""
        return self.model_copy(update={"directories": self.directories.anchored()})

    def overlaid(self, path: Path) -> Path:
        """Where furu writes what belongs at ``path``: in the overlay, if one is set."""
        if self.overlay is None:
            return path
        path = Path(os.path.normpath(path.absolute()))
        if path.is_relative_to(self.overlay) or self.overlay.is_relative_to(path):
            raise ValueError(
                f"the overlay {self.overlay} and {path}, which it stands for, "
                "must not hold one another: furu could not tell what it "
                "wrote in the debug session from what it only reads. Put the "
                "overlay in a folder of its own."
            )
        return self.overlay / path.relative_to(path.anchor)

    def layers(self, path: Path) -> tuple[Path, ...]:
        """Where furu reads what it writes at ``path``, first to last.

        That is ``path`` and then, when it is in the overlay, the path it
        stands for, which is only ever read.
        """
        if self.overlay is None or not path.is_relative_to(self.overlay):
            return (path,)
        return path, Path(path.anchor) / path.relative_to(self.overlay)

    @property
    def write_directories(self) -> _FuruDirectories:
        """The absolute directories furu writes to: in the overlay, if one is set."""
        return _FuruDirectories(
            **{name: self.overlaid(path) for name, path in self.directories.anchored()}
        )

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
