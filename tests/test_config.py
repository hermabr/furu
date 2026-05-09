from pathlib import Path
from typing import Any, cast

from furu.config import _FuruConfig, _FuruDirectories


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


def test_config_validates_directory_assignment() -> None:
    config = _FuruConfig()

    config.directories = cast(Any, {"data": "/tmp/assigned-furu-data"})

    assert config.directories == _FuruDirectories(data=Path("/tmp/assigned-furu-data"))
