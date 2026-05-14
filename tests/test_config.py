from pathlib import Path

import pytest
from pydantic import ValidationError

import furu.config as furu_config
from furu.config import (
    _FuruConfig,
    _FuruDirectories,
)


def test_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("FURU_DEBUG_MODE", "true")
    monkeypatch.setenv("FURU_DIRECTORIES__DATA", "/tmp/furu-data")

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(data=Path("/tmp/furu-data"))


def test_config_reads_pyproject_toml(tmp_path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu]
debug_mode = true

[tool.furu.directories]
data = "/tmp/furu-pyproject-data"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(data=Path("/tmp/furu-pyproject-data"))


def test_config_discovers_pyproject_toml_in_parent_directory(
    tmp_path, monkeypatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu.directories]
data = "/tmp/furu-parent-pyproject-data"
""",
        encoding="utf-8",
    )
    nested_directory = tmp_path / "src" / "project"
    nested_directory.mkdir(parents=True)
    monkeypatch.chdir(nested_directory)

    config = _FuruConfig()

    assert config.directories == _FuruDirectories(
        data=Path("/tmp/furu-parent-pyproject-data")
    )


def test_environment_overrides_pyproject_toml(tmp_path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu]
debug_mode = false

[tool.furu.directories]
data = "/tmp/furu-pyproject-data"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FURU_DEBUG_MODE", "true")
    monkeypatch.setenv("FURU_DIRECTORIES__DATA", "/tmp/furu-env-data")

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(data=Path("/tmp/furu-env-data"))


def test_config_is_frozen() -> None:
    config = _FuruConfig()

    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.directories = _FuruDirectories(data=Path("/tmp/assigned-furu-data"))


def test_config_module_value_can_be_replaced(monkeypatch) -> None:
    replacement_config = _FuruConfig(
        directories=_FuruDirectories(data=Path("/tmp/context-furu-data")),
    )

    monkeypatch.setattr(furu_config, "config", replacement_config)

    assert furu_config.config is replacement_config
