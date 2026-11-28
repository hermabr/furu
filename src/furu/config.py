import functools
import os
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)

from furu.utils import nfs_safe_unique_name

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


OVERLAY_ENV = "FURU_OVERLAY"
# In the overlay's root: says the folder is furu's, and keeps one inside a git
# repository out of `git status`.
_OVERLAY_MARK = (
    f"# A furu overlay ({OVERLAY_ENV}). Deleting this folder undoes its session.\n*\n"
)


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
        """Absolute, and under the overlay when one is set."""
        return _FuruDirectories(
            **{
                name: upper(path if path.is_absolute() else _project_anchor() / path)
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
        env_ignore_empty=True,
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
    # A debug session: every write goes under this folder and the real store
    # is only read (see `upper`). Only FURU_OVERLAY in the environment sets it,
    # so no checkout or file can turn one on for whoever runs there next.
    overlay: Path | None = None
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories)
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)

    @property
    def manager_version_dir(self) -> Path:
        """This furu version's manager: one per version, so scripts never meet another."""
        return upper(self.manager_dir) / version("furu")

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
        pyproject = PyprojectTomlConfigSettingsSource(settings_cls)
        for file, source in ((".env", dotenv_settings), ("pyproject.toml", pyproject)):
            if "overlay" in source():
                raise ValueError(
                    f"{file} sets a furu overlay, which only the environment "
                    f"may: run with {OVERLAY_ENV}=/an/absolute/folder"
                )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            pyproject,
            file_secret_settings,
        )


_config = _Config()


def get_config() -> _Config:
    return _config


def _set_config(config: _Config) -> None:
    global _config
    _config = config


@functools.cache
def _claim(overlay: Path) -> Path:
    """Make ``overlay`` this session's, once per process: a new or empty folder,
    or one an earlier session claimed. Any other folder is somebody's files."""
    if not overlay.is_absolute():
        raise ValueError(f"{OVERLAY_ENV}={overlay} must be an absolute path")
    mark = overlay / ".gitignore"

    def claimed() -> bool:
        try:
            return mark.read_text(encoding="utf-8") == _OVERLAY_MARK
        except OSError:
            return False

    if not claimed():
        # Made whole beside the folder and renamed onto it, which works only
        # while that is missing or empty, whoever else claims it now.
        seed = nfs_safe_unique_name(overlay, name="tmp")
        seed.mkdir(parents=True)
        (seed / mark.name).write_text(_OVERLAY_MARK, encoding="utf-8")
        try:
            os.rename(seed, overlay)
        except OSError:
            shutil.rmtree(seed, ignore_errors=True)
        if not claimed():
            raise RuntimeError(
                f"{OVERLAY_ENV}={overlay} exists and is not a furu overlay. An "
                "overlay is a new or empty folder, or one an earlier session made."
            )
    return overlay


def upper(path: Path) -> Path:
    """Where ``path`` is written: mirrored under the overlay when one is set.

    Everything furu writes goes through here, so a session with an overlay
    leaves the real store as it was. `layers` is where ``path`` is read.
    """
    if (overlay := _config.overlay) is None:
        return path
    overlay = _claim(Path(os.path.normpath(overlay)))
    path = Path(os.path.abspath(path))
    if path.is_relative_to(overlay):
        return path
    return overlay / path.relative_to(path.anchor)


def lower(path: Path) -> Path:
    """The real store's ``path``, for one that `upper` put under the overlay."""
    if (overlay := _config.overlay) is None:
        return path
    overlay = Path(os.path.normpath(overlay))
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(overlay):
        return path
    return Path(path.anchor) / path.relative_to(overlay)


def layers(path: Path) -> list[Path]:
    """Where ``path`` is read: under the overlay first, then in the real store."""
    return list(dict.fromkeys((upper(path), lower(path))))
