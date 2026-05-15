from pathlib import Path
from typing import Self

from pydantic import ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)


class _FuruDirectories(BaseSettings):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objects: Path
    executions: Path
    # TODO: make this better and more user configurable, so that it is easy for the user to define exactly which paths they want to save to and which furu objects should save where

    @classmethod
    def default(cls) -> Self:
        # TODO: make sure this location is deterministic/more predictable, such as by finding the next .git or pyproject.toml or furu directory
        base_dir = Path("furu")
        objects_dir = base_dir / "objects"
        executions_dir = base_dir / "executions"
        return cls(objects=objects_dir, executions=executions_dir)


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


_config = _FuruConfig()


def get_config() -> _FuruConfig:
    return _config
