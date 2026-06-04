import os
import tomllib
from pathlib import Path
from typing import Any
from typing import Self

from pydantic import ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)

_WORKER_JSON_CONFIG_FILE_ENV_VAR = "_FURU_WORKER_JSON_CONFIG_FILE"


class _PyprojectTableConfigSettingsSource(InitSettingsSource):
    def __init__(
        self,
        settings_cls: type[BaseSettings],
        *,
        table_header: tuple[str, ...],
    ) -> None:
        pyproject_path = PyprojectTomlConfigSettingsSource._pick_pyproject_toml_file(
            None,
            settings_cls.model_config.get("pyproject_toml_depth", 0),
        )
        data: Any = {}
        if pyproject_path.is_file():
            with pyproject_path.open("rb") as file:
                data = tomllib.load(file)
            for key in table_header:
                if not isinstance(data, dict):
                    data = {}
                    break
                data = data.get(key, {})
        super().__init__(settings_cls, data if isinstance(data, dict) else {})


class _FuruDirectories(BaseSettings):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objects: Path
    executions: Path

    @classmethod
    def default(cls) -> Self:
        # TODO: make sure this location is deterministic/more predictable, such as by finding the next .git or pyproject.toml or furu directory
        base_dir = Path("furu")
        objects_dir = base_dir / "objects"
        executions_dir = base_dir / "executions"
        return cls(objects=objects_dir, executions=executions_dir)


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
    codec: tuple[str, ...] = ()
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories.default)
    worker: _FuruWorkerConfig = Field(default_factory=_FuruWorkerConfig)

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
            _PyprojectTableConfigSettingsSource(
                settings_cls,
                table_header=("furu",),
            ),
            PyprojectTomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


_config = _FuruConfig()


def get_config() -> _FuruConfig:
    return _config
