import functools
import os
import subprocess
from contextlib import suppress
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
        """Absolute, and inside the overlay in an overlay session."""
        absolute = {
            name: path if path.is_absolute() else _project_anchor() / path
            for name, path in self
        }
        return _FuruDirectories(
            **{name: layers(path)[0] for name, path in absolute.items()}
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
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)
    # FURU_OVERLAY only: see layers().
    overlay: Path | None = None

    @field_validator("overlay")
    @classmethod
    def _absolute(cls, overlay: Path | None) -> Path | None:
        if overlay is not None and not overlay.is_absolute():
            raise ValueError(f"FURU_OVERLAY must be an absolute path, got {overlay}")
        return overlay if overlay is None else Path(os.path.normpath(overlay))

    @property
    def manager_version_dir(self) -> Path:
        """This furu version's manager: one per version, so scripts never meet
        another, and one per overlay, so an overlay session never meets the
        store's."""
        return layers(self.manager_dir / version("furu"))[0]

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
            _Pyproject(settings_cls),
            file_secret_settings,
        )


class _Pyproject(PyprojectTomlConfigSettingsSource):
    def __call__(self) -> dict[str, object]:
        settings = super().__call__()
        if "overlay" in settings:
            raise ValueError(
                "[tool.furu] sets overlay, which only the FURU_OVERLAY "
                "environment variable may: a committed overlay would hide every "
                "launch's results from the store"
            )
        return settings


_config = _Config()


def get_config() -> _Config:
    return _config


def _set_config(config: _Config) -> None:
    global _config
    _config = config


# Marks a folder as an overlay, and keeps it out of `git status`.
_OVERLAY_MARK = "# a furu overlay: rm -rf this folder to undo its session\n*\n"


def layers(path: Path) -> list[Path]:
    """Where what furu keeps at ``path`` is: in the overlay, then in the store.

    With ``FURU_OVERLAY`` set, furu writes nothing but the overlay: what it
    would write to an absolute path goes to that path under the overlay, and
    what it reads it looks for there first. The store is read, never written
    or locked, so ``rm -rf`` of the overlay undoes the session. The mapping
    goes both ways, so either path gives both layers. Without an overlay,
    ``path`` is the only layer.
    """
    if (overlay := get_config().overlay) is None:
        return [path]
    path = Path(os.path.abspath(path))
    store = (
        Path("/", path.relative_to(overlay)) if path.is_relative_to(overlay) else path
    )
    return [_claimed(overlay) / store.relative_to(store.anchor), store]


@functools.cache
def _claimed(overlay: Path) -> Path:
    """The overlay, made the first time: a folder of furu's own, never one in use."""
    mark = overlay / ".gitignore"
    with suppress(OSError):
        if mark.read_text(encoding="utf-8") == _OVERLAY_MARK:
            return overlay
    with suppress(FileNotFoundError):
        if any(overlay.iterdir()):
            raise RuntimeError(
                f"FURU_OVERLAY={overlay} is a folder that furu did not make. An "
                "overlay is undone with rm -rf, so point it at a new folder."
            )
    overlay.mkdir(parents=True, exist_ok=True)
    mark.write_text(_OVERLAY_MARK, encoding="utf-8")
    return overlay
