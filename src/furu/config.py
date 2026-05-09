from pathlib import Path
from typing import Self

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _FuruDirectories(BaseSettings):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data: Path
    # TODO: make this better and more user configurable, so that it is easy for the user to define exactly which paths they want to save to and which furu objects should save where

    @classmethod
    def default(cls) -> Self:
        # TODO: make sure this location is deterministic/more predictable, such as by finding the next .git or pyproject.toml or furu directory
        base_dir = Path("furu")
        data_dir = base_dir / "data"
        return cls(data=data_dir)


class _FuruConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FURU_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    debug_mode: bool = False
    directories: _FuruDirectories = Field(default_factory=_FuruDirectories.default)


config = _FuruConfig()
