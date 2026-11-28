"""Furu's settings, and the overlay folder of a debug session.

With ``FURU_OVERLAY=/abs/folder`` in the environment, furu runs against every
result in the store and changes nothing there: no file, no folder and no lock.
Whatever it would write at an absolute path it writes at the same path under
the folder (`over`), so no two storage roots meet and a path there says which
path it stands for (`under`). That covers results, also those of a class with
its own ``Metadata(storage=...)``, working folders, schema files, plans,
provenance, code snapshots and the manager's own files. Whatever it reads it
looks for under the folder first and in the store after that (`layers`).
Deleting the folder undoes the session.

A session has a manager of its own, found under the folder: a manager knows a
spec by its id, so one shared with other launches would hand a session's spec
to a worker that publishes into the store, or the other way round. Start it
with the same ``FURU_OVERLAY``. Workers on other machines must see the folder.
"""

import functools
import os
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
        """Where furu writes: absolute, and under the overlay in a debug session."""
        return _FuruDirectories(
            **{
                name: over(path if path.is_absolute() else _project_anchor() / path)
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
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories)
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)
    # The folder of a debug session, which the FURU_OVERLAY environment
    # variable asks for and no file does (see `_load`). A worker gets it with
    # the rest of its launch's settings.
    overlay: Path | None = None

    @property
    def manager_version_dir(self) -> Path:
        """This furu version's manager: one per version, and one per overlay,
        so scripts never meet another."""
        return over(self.manager_dir) / version("furu")

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


OVERLAY_ENV_VAR = "FURU_OVERLAY"


def _load() -> _Config:
    config = _Config()
    asked = os.environ.get(OVERLAY_ENV_VAR)
    if config.overlay != (Path(asked) if asked else None):
        raise RuntimeError(
            f"the overlay folder {config.overlay} is set in .env or under "
            "[tool.furu] in pyproject.toml. Only the environment asks for a "
            f"debug session, so that none is committed with the repo: remove "
            f"it there and run with {OVERLAY_ENV_VAR}={config.overlay}"
        )
    return config


_config = _load()


def get_config() -> _Config:
    return _config


def _set_config(config: _Config) -> None:
    global _config
    _config = config


# What the one file furu puts in a new overlay folder holds. It keeps a folder
# inside a repo out of `git status`, and says that the folder is an overlay.
_OVERLAY_MARK = (
    f"# A furu overlay ({OVERLAY_ENV_VAR}): deleting this folder undoes the debug "
    "session.\n*\n"
)


@functools.cache
def _claimed(root: Path) -> Path:
    """The overlay folder, once it is known to hold an overlay and nothing else.

    A new or empty folder becomes one. Any other folder is refused before
    anything is written into it: its files would be hidden from git, and
    deleting it would no longer only undo the session.
    """
    if not root.is_absolute():
        raise RuntimeError(
            f"{OVERLAY_ENV_VAR}={root} is not an absolute path, and workers do "
            "not run where this script does"
        )
    root.mkdir(parents=True, exist_ok=True)
    mark = root / ".gitignore"
    try:
        ours = mark.read_text(encoding="utf-8") == _OVERLAY_MARK
    except FileNotFoundError:
        if ours := not os.listdir(root):
            mark.write_text(_OVERLAY_MARK, encoding="utf-8")
    if not ours:
        raise RuntimeError(
            f"{OVERLAY_ENV_VAR}={root} holds files that are not a furu "
            "overlay's. A debug session takes a new or empty folder, which "
            "deleting then undoes, and furu writes nothing into this one."
        )
    return root


def over(path: Path) -> Path:
    """Where furu writes what it would write at ``path``.

    In a debug session that is the same path under the overlay folder, and
    ``path`` itself when it is there already. Without one it is ``path``.
    """
    if (root := _config.overlay) is None or path.is_relative_to(root):
        return path
    path = path.absolute()
    return _claimed(root) / path.relative_to(path.anchor)


def under(path: Path) -> Path:
    """The path in the store that ``path`` stands for, if it is under the overlay folder."""
    if (root := _config.overlay) is None or not path.is_relative_to(root):
        return path
    return Path(root.anchor, path.relative_to(root))


def layers(path: Path) -> list[Path]:
    """Where furu reads what it writes at ``path``.

    In a debug session the overlay's is first and wins, and the store's own,
    which is never written to, comes after it.
    """
    return [path] if _config.overlay is None else [over(path), under(path)]
