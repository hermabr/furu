import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)

_WORKER_JSON_CONFIG_FILE_ENV_VAR = "_FURU_WORKER_JSON_CONFIG_FILE"


class _FuruDirectories(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objects: Path = Path("furu") / "objects"
    executions: Path = Path("furu") / "executions"
    debug: Path = Path("furu") / "debug"


class _FuruWorkerConfig(BaseModel):
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
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories)
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)

    @property
    def run_directories(self) -> _FuruDirectories:
        if self.debug_mode:
            return _FuruDirectories(
                objects=self.directories.debug / "objects",
                executions=self.directories.debug / "executions",
                debug=self.directories.debug,
            )
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


def _set_config(config: _FuruConfig) -> None:
    global _config
    _config = config
