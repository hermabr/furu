import os
from pathlib import Path
from typing import Self

from pydantic import ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)

_WORKER_JSON_CONFIG_FILE_ENV_VAR = "_FURU_WORKER_JSON_CONFIG_FILE"


class _FuruDirectories(BaseSettings):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objects: Path
    executions: Path
    debug: Path = Path("furu") / "debug"

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> Self:
        return cls(
            objects=base_dir / "objects",
            executions=base_dir / "executions",
            debug=base_dir / "debug",
        )

    @classmethod
    def from_debug_dir(cls, debug_dir: Path) -> Self:
        return cls(
            objects=debug_dir / "objects",
            executions=debug_dir / "executions",
            debug=debug_dir,
        )

    @classmethod
    def default(cls) -> Self:
        # TODO: make sure this location is deterministic/more predictable, such as by finding the next .git or pyproject.toml or furu directory
        return cls.from_base_dir(Path("furu"))


class _FuruWorkerConfig(BaseSettings):
    model_config = ConfigDict(extra="forbid", frozen=True)

    idle_timeout_seconds: float = 60.0
    max_failed_restarts: int = 16
    max_retries_per_object: int = 3


class _FuruConfig(BaseSettings):
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
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories.default)
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)

    @property
    def run_directories(self) -> _FuruDirectories:
        if self.debug_mode:
            return _FuruDirectories.from_debug_dir(self.directories.debug)
        return self.directories

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


_config = _FuruConfig()


def get_config() -> _FuruConfig:
    return _config
