from pathlib import Path

import pytest
from pydantic import ValidationError

import furu.config as furu_config
from furu.config import (
    _FuruConfig,
    _FuruDirectories,
    get_config,
)


def test_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("FURU_DEBUG_MODE", "true")
    monkeypatch.setenv("FURU_DIRECTORIES__DATA", "/tmp/furu-data")
    monkeypatch.setenv("FURU_DIRECTORIES__EXECUTIONS", "/tmp/furu-executions")

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(
        data=Path("/tmp/furu-data"),
        executions=Path("/tmp/furu-executions"),
    )


def test_config_reads_pyproject_toml(tmp_path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu]
debug_mode = true

[tool.furu.directories]
data = "/tmp/furu-pyproject-data"
executions = "/tmp/furu-pyproject-executions"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(
        data=Path("/tmp/furu-pyproject-data"),
        executions=Path("/tmp/furu-pyproject-executions"),
    )


def test_config_discovers_pyproject_toml_in_parent_directory(
    tmp_path, monkeypatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu.directories]
data = "/tmp/furu-parent-pyproject-data"
executions = "/tmp/furu-parent-pyproject-executions"
""",
        encoding="utf-8",
    )
    nested_directory = tmp_path / "src" / "project"
    nested_directory.mkdir(parents=True)
    monkeypatch.chdir(nested_directory)

    config = _FuruConfig()

    assert config.directories == _FuruDirectories(
        data=Path("/tmp/furu-parent-pyproject-data"),
        executions=Path("/tmp/furu-parent-pyproject-executions"),
    )


def test_environment_overrides_pyproject_toml(tmp_path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu]
debug_mode = false

[tool.furu.directories]
data = "/tmp/furu-pyproject-data"
executions = "/tmp/furu-pyproject-executions"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FURU_DEBUG_MODE", "true")
    monkeypatch.setenv("FURU_DIRECTORIES__DATA", "/tmp/furu-env-data")
    monkeypatch.setenv("FURU_DIRECTORIES__EXECUTIONS", "/tmp/furu-env-executions")

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(
        data=Path("/tmp/furu-env-data"),
        executions=Path("/tmp/furu-env-executions"),
    )


def test_config_is_frozen() -> None:
    config = _FuruConfig()

    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.directories = _FuruDirectories(
            data=Path("/tmp/assigned-furu-data"),
            executions=Path("/tmp/assigned-furu-executions"),
        )


def test_private_config_module_value_can_be_replaced(monkeypatch) -> None:
    replacement_config = _FuruConfig(
        directories=_FuruDirectories(
            data=Path("/tmp/context-furu-data"),
            executions=Path("/tmp/context-furu-executions"),
        ),
    )

    monkeypatch.setattr(furu_config, "_config", replacement_config)

    assert get_config() is replacement_config
