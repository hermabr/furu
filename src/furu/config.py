import contextlib
import functools
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


def _same_kind_overlap(a: Path, b: Path) -> bool:
    """Whether two paths (both absolute or both relative) overlap."""
    if a.is_absolute():
        a, b = a.resolve(), b.resolve()
    else:
        a, b = Path(os.path.normpath(a)), Path(os.path.normpath(b))
    return a.is_relative_to(b) or b.is_relative_to(a)


def _reject_overlap(overlay: Path, target: Path, *, label: str) -> None:
    if _same_kind_overlap(overlay, target):
        raise ValueError(
            f"overlay {overlay} cannot overlap {label} ({target}): "
            "writes would touch the base store or deleting the overlay would remove it"
        )


def _reject_repo_root(overlay: Path, repo_root: Path) -> None:
    if repo_root.is_absolute() and repo_root.resolve().is_relative_to(
        overlay.resolve()
    ):
        raise ValueError(
            f"overlay {overlay} cannot be or contain the repository root {repo_root}"
        )


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
    # A debug session (FURU_OVERLAY=<folder>): everything furu writes goes
    # under this folder, at the absolute path it would otherwise have, while
    # results at the real paths still load read-only (never written, created,
    # or locked). Deleting this folder undoes the session. A relative overlay
    # anchors at the main worktree root, like directories.
    overlay: Path | None = None
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)

    @field_validator("overlay", mode="before")
    @classmethod
    def _empty_overlay_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("overlay")
    @classmethod
    def _normalize_overlay(cls, overlay: Path | None) -> Path | None:
        if overlay is None:
            return None
        normalized = Path(os.path.normpath(overlay))
        if normalized == Path(".") or (
            normalized.is_absolute() and normalized == Path(normalized.anchor)
        ):
            raise ValueError(
                f"overlay cannot be the filesystem or repository root: {overlay}"
            )
        return normalized

    @model_validator(mode="after")
    def _validate_overlay(self) -> "_Config":
        if self.overlay is None:
            return self
        for name, directory in self.directories:
            if self.overlay.is_absolute() == directory.is_absolute():
                _reject_overlap(self.overlay, directory, label=f"directories.{name}")
        if self.overlay.is_absolute():
            with contextlib.suppress(RuntimeError):
                _reject_repo_root(self.overlay, _project_anchor())
            with contextlib.suppress(RuntimeError):
                for name, directory in self.directories.anchored():
                    _reject_overlap(
                        self.overlay, directory, label=f"directories.{name}"
                    )
        return self

    @property
    def manager_version_dir(self) -> Path:
        """This furu version's manager: one per version, so scripts never meet another."""
        return self.manager_dir / version("furu")

    def _overlay(self) -> Path | None:
        """The anchored, normalized, validated overlay root, or None."""
        if self.overlay is None:
            return None
        overlay = Path(os.path.normpath(self.overlay))
        if overlay == Path("."):
            raise ValueError(f"overlay cannot be the repository root: {self.overlay}")
        if not overlay.is_absolute():
            overlay = Path(os.path.normpath(_project_anchor() / overlay))
        if overlay == Path(overlay.anchor):
            raise ValueError(f"overlay cannot be the filesystem root: {self.overlay}")
        with contextlib.suppress(RuntimeError):
            _reject_repo_root(overlay, _project_anchor())
        for name, directory in self.directories:
            if directory.is_absolute():
                _reject_overlap(overlay, directory, label=f"directories.{name}")
        with contextlib.suppress(RuntimeError):
            for name, directory in self.directories.anchored():
                _reject_overlap(overlay, directory, label=f"directories.{name}")
        return overlay

    def anchored(self) -> "_Config":
        """This config with absolute directories, for processes whose working
        directory (e.g. a snapshot) must not decide where data lives."""
        return self.model_copy(
            update={
                "directories": self.directories.anchored(),
                "overlay": self._overlay(),
            }
        )

    def overlaid(self, path: Path) -> Path:
        """Where furu writes what belongs at ``path``: under the overlay, if one is set."""
        overlay = self._overlay()
        if overlay is None:
            return path
        target = Path(os.path.normpath(path.absolute()))
        _reject_overlap(overlay, target, label="storage")
        return overlay / target.relative_to(target.anchor)

    def layers(self, path: Path) -> tuple[Path, ...]:
        """Where furu reads what belongs at ``path``: overlay first, then base."""
        overlay = self._overlay()
        if overlay is None:
            return (path,)
        norm = Path(os.path.normpath(path.absolute()))
        if norm == overlay or not norm.is_relative_to(overlay):
            return (path,)
        return norm, Path(norm.anchor) / norm.relative_to(overlay)

    @property
    def write_directories(self) -> _FuruDirectories:
        """The absolute directories furu writes to: in the overlay, if one is set."""
        anchored = self.directories.anchored()
        if self.overlay is None:
            return anchored
        return _FuruDirectories(
            **{name: self.overlaid(path) for name, path in anchored}
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
