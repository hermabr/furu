from pathlib import Path

import pytest
from pydantic import ValidationError

import furu.config as furu_config
from furu.config import (
    _FuruConfig,
    _FuruDirectories,
    _FuruResultConfig,
    _FuruWorkerConfig,
    _WORKER_JSON_CONFIG_FILE_ENV_VAR,
    get_config,
)


def test_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("FURU_DEBUG_MODE", "true")
    monkeypatch.setenv("FURU_DIRECTORIES__OBJECTS", "/tmp/furu-objects")
    monkeypatch.setenv("FURU_DIRECTORIES__EXECUTIONS", "/tmp/furu-executions")
    monkeypatch.setenv(
        "FURU_RESULT__CODECS",
        '["env_lib.FirstCodec", "env_lib.SecondCodec"]',
    )
    monkeypatch.setenv(
        "FURU_SERIALIZERS",
        '["env_lib.FirstSerializer", "env_lib.SecondSerializer"]',
    )
    monkeypatch.setenv("FURU_WORKER__IDLE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("FURU_WORKER__MAX_FAILED_RESTARTS", "7")
    monkeypatch.setenv("FURU_WORKER__MAX_RETRIES_PER_OBJECT", "3")

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(
        objects=Path("/tmp/furu-objects"),
        executions=Path("/tmp/furu-executions"),
    )
    assert config.result == _FuruResultConfig(
        codecs=("env_lib.FirstCodec", "env_lib.SecondCodec")
    )
    assert config.serializers == (
        "env_lib.FirstSerializer",
        "env_lib.SecondSerializer",
    )
    assert config.worker == _FuruWorkerConfig(
        idle_timeout_seconds=12.5,
        max_failed_restarts=7,
        max_retries_per_object=3,
    )


def test_config_reads_pyproject_toml(tmp_path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu]
debug_mode = true
serializers = ["my_lib.SomeSerializer"]

[tool.furu.directories]
objects = "/tmp/furu-pyproject-objects"
executions = "/tmp/furu-pyproject-executions"

[tool.furu.result]
codecs = ["my_lib.my_fn.SomeCodec"]

[tool.furu.worker]
idle_timeout_seconds = 7.5
max_failed_restarts = 7
max_retries_per_object = 3
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(
        objects=Path("/tmp/furu-pyproject-objects"),
        executions=Path("/tmp/furu-pyproject-executions"),
    )
    assert config.result == _FuruResultConfig(codecs=("my_lib.my_fn.SomeCodec",))
    assert config.serializers == ("my_lib.SomeSerializer",)
    assert config.worker == _FuruWorkerConfig(
        idle_timeout_seconds=7.5,
        max_failed_restarts=7,
        max_retries_per_object=3,
    )


def test_config_discovers_pyproject_toml_in_parent_directory(
    tmp_path, monkeypatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu.directories]
objects = "/tmp/furu-parent-pyproject-objects"
executions = "/tmp/furu-parent-pyproject-executions"
""",
        encoding="utf-8",
    )
    nested_directory = tmp_path / "src" / "project"
    nested_directory.mkdir(parents=True)
    monkeypatch.chdir(nested_directory)

    config = _FuruConfig()

    assert config.directories == _FuruDirectories(
        objects=Path("/tmp/furu-parent-pyproject-objects"),
        executions=Path("/tmp/furu-parent-pyproject-executions"),
    )


def test_environment_overrides_pyproject_toml(tmp_path, monkeypatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.furu]
debug_mode = false

[tool.furu.directories]
objects = "/tmp/furu-pyproject-objects"
executions = "/tmp/furu-pyproject-executions"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FURU_DEBUG_MODE", "true")
    monkeypatch.setenv("FURU_DIRECTORIES__OBJECTS", "/tmp/furu-env-objects")
    monkeypatch.setenv("FURU_DIRECTORIES__EXECUTIONS", "/tmp/furu-env-executions")

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(
        objects=Path("/tmp/furu-env-objects"),
        executions=Path("/tmp/furu-env-executions"),
    )


def test_config_reads_json_config_file(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "furu-config.json"
    config_file.write_text(
        """
{
  "debug_mode": true,
  "serializers": ["json_lib.SomeSerializer"],
  "directories": {
    "objects": "/tmp/furu-json-objects",
    "executions": "/tmp/furu-json-executions"
  },
  "result": {
    "codecs": ["json_lib.SomeCodec"]
  },
  "worker": {
    "idle_timeout_seconds": 9.5,
    "max_failed_restarts": 7,
    "max_retries_per_object": 3
  }
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(_WORKER_JSON_CONFIG_FILE_ENV_VAR, str(config_file))
    monkeypatch.setenv("FURU_DEBUG_MODE", "false")
    monkeypatch.setenv("FURU_WORKER__IDLE_TIMEOUT_SECONDS", "12.5")

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(
        objects=Path("/tmp/furu-json-objects"),
        executions=Path("/tmp/furu-json-executions"),
    )
    assert config.result == _FuruResultConfig(codecs=("json_lib.SomeCodec",))
    assert config.serializers == ("json_lib.SomeSerializer",)
    assert config.worker == _FuruWorkerConfig(
        idle_timeout_seconds=9.5,
        max_failed_restarts=7,
        max_retries_per_object=3,
    )


def test_config_is_frozen() -> None:
    config = _FuruConfig()

    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.directories = _FuruDirectories(
            objects=Path("/tmp/assigned-furu-objects"),
            executions=Path("/tmp/assigned-furu-executions"),
        )


def test_private_config_module_value_can_be_replaced(monkeypatch) -> None:
    replacement_config = _FuruConfig(
        directories=_FuruDirectories(
            objects=Path("/tmp/context-furu-objects"),
            executions=Path("/tmp/context-furu-executions"),
        ),
    )

    monkeypatch.setattr(furu_config, "_config", replacement_config)

    assert get_config() is replacement_config
