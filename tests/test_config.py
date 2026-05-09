from pathlib import Path
from typing import Any, cast

from furu.config import _FuruConfig, _FuruDirectories


def test_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("FURU_DEBUG_MODE", "true")
    monkeypatch.setenv("FURU_DIRECTORIES__DATA", "/tmp/furu-data")

    config = _FuruConfig()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(data=Path("/tmp/furu-data"))


def test_config_validates_directory_assignment() -> None:
    config = _FuruConfig()

    config.directories = cast(Any, {"data": "/tmp/assigned-furu-data"})

    assert config.directories == _FuruDirectories(data=Path("/tmp/assigned-furu-data"))
