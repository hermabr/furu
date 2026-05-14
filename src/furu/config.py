from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Self

from pydantic import ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)


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


_CONFIG: ContextVar[_FuruConfig] = ContextVar("furu_config", default=_FuruConfig())


def get_config() -> _FuruConfig:
    return _CONFIG.get()


def set_config(config: _FuruConfig) -> Token[_FuruConfig]:
    return _CONFIG.set(config)


def reset_config(token: Token[_FuruConfig]) -> None:
    _CONFIG.reset(token)


@contextmanager
def use_config(config: _FuruConfig) -> Iterator[_FuruConfig]:
    token = set_config(config)
    try:
        yield config
    finally:
        reset_config(token)


def replace_config(config: _FuruConfig | None = None, **updates: Any) -> _FuruConfig:
    data = (get_config() if config is None else config).model_dump()
    data.update(updates)
    return _FuruConfig.model_validate(data)


class _ActiveConfigProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_config(), name)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError(
            "Furu config is frozen; use use_config() with a replacement _FuruConfig"
        )

    def __repr__(self) -> str:
        return repr(get_config())


config = _ActiveConfigProxy()
