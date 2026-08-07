import functools
import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ByteSize, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsError,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_WORKER_JSON_CONFIG_FILE_ENV_VAR = "_FURU_WORKER_JSON_CONFIG_FILE"
_GLOBAL_CONFIG_KEYS = {
    "worker": {
        "connect_host",
        "idle_timeout_seconds",
        "max_failed_restarts",
        "max_retries_per_object",
    },
    "provenance": {"max_snapshot_bytes"},
}


class _GlobalTomlConfigSettingsSource(TomlConfigSettingsSource):
    def __call__(self) -> dict[str, Any]:
        values = super().__call__()
        unsupported = set(values) - _GLOBAL_CONFIG_KEYS.keys()
        for section, allowed in _GLOBAL_CONFIG_KEYS.items():
            if isinstance(section_values := values.get(section), dict):
                unsupported |= {
                    f"{section}.{key}" for key in section_values if key not in allowed
                }
        if unsupported:
            raise SettingsError(
                f"Unsupported global Furu setting(s): {', '.join(sorted(unsupported))}"
            )
        return values


@functools.cache
def _project_anchor() -> Path:
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return Path(common_dir).parent
    except (OSError, subprocess.CalledProcessError):
        pass
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise RuntimeError(
        f"no git repository or pyproject.toml found from {cwd} upward.\n"
        "furu anchors its data directories to the project root. Create one with:\n"
        "  uv init"
    )


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
    max_retries_per_object: int = 3


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
            _GlobalTomlConfigSettingsSource(
                settings_cls,
                Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser()
                / "furu"
                / "furu.toml",
            ),
            file_secret_settings,
        )


_config = _Config()


def get_config() -> _Config:
    return _config


def _set_config(config: _Config) -> None:
    global _config
    _config = config
