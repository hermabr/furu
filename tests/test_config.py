import subprocess
import sys
from pathlib import Path

import pytest

from furu.config import (
    _WORKER_JSON_CONFIG_FILE_ENV_VAR,
    _Config,
    _FuruDirectories,
    _FuruWorkerConfig,
    _project_anchor,
    get_config,
)
from furu.testing import override_config

_PYPROJECT_TOML = """
[tool.furu]
debug_mode = true

[tool.furu.directories]
objects = "/tmp/furu-pyproject-objects"
executions = "/tmp/furu-pyproject-executions"
debug = "/tmp/furu-pyproject-debug"

[tool.furu.worker]
idle_timeout_seconds = 7.5
max_retries_per_object = 3
"""

_DEBUG_OFF_PYPROJECT_TOML = """
[tool.furu]
debug_mode = false

[tool.furu.directories]
objects = "/tmp/furu-pyproject-objects"
executions = "/tmp/furu-pyproject-executions"
debug = "/tmp/furu-pyproject-debug"
"""

_WORKER_JSON = """
{
  "coordinator_url": "ws://furu:secret@coordinator.test:1",
  "debug_mode": true,
  "directories": {
    "objects": "/tmp/furu-json-objects",
    "executions": "/tmp/furu-json-executions",
    "debug": "/tmp/furu-json-debug"
  },
  "worker": {
    "idle_timeout_seconds": 9.5,
    "max_retries_per_object": 3
  }
}
"""


def _debug_env(prefix: str) -> dict[str, str]:
    return {
        "FURU_DEBUG_MODE": "true",
        "FURU_DIRECTORIES__OBJECTS": f"{prefix}-objects",
        "FURU_DIRECTORIES__EXECUTIONS": f"{prefix}-executions",
        "FURU_DIRECTORIES__DEBUG": f"{prefix}-debug",
    }


@pytest.mark.parametrize(
    ("files", "env", "prefix", "worker"),
    [
        pytest.param(
            {},
            {
                **_debug_env("/tmp/furu"),
                "FURU_WORKER__CONNECT_HOST": "login01.cluster",
                "FURU_WORKER__IDLE_TIMEOUT_SECONDS": "12.5",
                "FURU_WORKER__MAX_RETRIES_PER_OBJECT": "3",
            },
            "/tmp/furu",
            _FuruWorkerConfig(
                connect_host="login01.cluster",
                idle_timeout_seconds=12.5,
                max_retries_per_object=3,
            ),
            id="environment",
        ),
        pytest.param(
            {"pyproject.toml": _PYPROJECT_TOML},
            {},
            "/tmp/furu-pyproject",
            _FuruWorkerConfig(idle_timeout_seconds=7.5, max_retries_per_object=3),
            id="pyproject-toml-in-a-parent-directory",
        ),
        pytest.param(
            {"pyproject.toml": _DEBUG_OFF_PYPROJECT_TOML},
            _debug_env("/tmp/furu-env"),
            "/tmp/furu-env",
            _FuruWorkerConfig(),
            id="environment-overrides-pyproject-toml",
        ),
        pytest.param(
            {"worker.config.json": _WORKER_JSON},
            {"FURU_DEBUG_MODE": "false", "FURU_WORKER__IDLE_TIMEOUT_SECONDS": "12.5"},
            "/tmp/furu-json",
            _FuruWorkerConfig(idle_timeout_seconds=9.5, max_retries_per_object=3),
            id="worker-json-overrides-environment",
        ),
    ],
)
def test_config_sources(
    files: dict[str, str],
    env: dict[str, str],
    prefix: str,
    worker: _FuruWorkerConfig,
    tmp_path,
    monkeypatch,
) -> None:
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    nested = tmp_path / "src" / "project"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    # Only the worker-json case writes this file; a missing file adds nothing.
    monkeypatch.setenv(
        _WORKER_JSON_CONFIG_FILE_ENV_VAR, str(tmp_path / "worker.config.json")
    )
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    config = _Config()

    assert config.debug_mode is True
    assert config.directories == _FuruDirectories(
        objects=Path(f"{prefix}-objects"),
        executions=Path(f"{prefix}-executions"),
        debug=Path(f"{prefix}-debug"),
    )
    debug = Path(f"{prefix}-debug")
    assert config.run_directories == _FuruDirectories(
        objects=debug / "objects",
        executions=debug / "executions",
        snapshots=debug / "snapshots",
        debug=debug,
    )
    assert config.worker == worker


def test_debug_mode_uses_default_debug_directory(monkeypatch) -> None:
    monkeypatch.setenv("FURU_DEBUG_MODE", "true")
    monkeypatch.setattr("furu.config._project_anchor", lambda: Path())

    config = _Config()

    assert config.debug_mode is True
    assert config.directories.debug == Path("furu-data") / "debug"
    assert config.run_directories == _FuruDirectories(
        objects=Path("furu-data") / "debug" / "objects",
        executions=Path("furu-data") / "debug" / "executions",
        snapshots=Path("furu-data") / "debug" / "snapshots",
        debug=Path("furu-data") / "debug",
    )


def test_relative_directories_anchor_to_main_worktree(tmp_path, monkeypatch) -> None:
    main = tmp_path / "main"
    main.mkdir()
    (main / "pyproject.toml").write_text("", encoding="utf-8")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], cwd=main, check=True)
    subprocess.run([*git, "add", "pyproject.toml"], cwd=main, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=main, check=True)
    subprocess.run(
        [*git, "worktree", "add", "-q", str(tmp_path / "linked")], cwd=main, check=True
    )

    # The linked worktree has its own pyproject.toml, but the git worktree
    # mapping wins so all worktrees share the main checkout's directories.
    monkeypatch.chdir(tmp_path / "linked")
    _project_anchor.cache_clear()
    try:
        expected = main.resolve() / "furu-data" / "objects"
        assert _Config().run_directories.objects == expected

        monkeypatch.chdir(main)
        _project_anchor.cache_clear()
        assert _Config().run_directories.objects == expected
    finally:
        _project_anchor.cache_clear()


def test_relative_directories_anchor_to_pyproject_root(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)
    _project_anchor.cache_clear()
    try:
        expected = tmp_path.resolve() / "furu-data" / "objects"
        assert _Config().run_directories.objects == expected
    finally:
        _project_anchor.cache_clear()


def test_import_outside_project_defers_anchor_crash(tmp_path) -> None:
    code = (
        "import furu\n"
        "from furu.config import get_config\n"
        "config = get_config()\n"
        "try:\n"
        "    config.run_directories\n"
        "except RuntimeError as error:\n"
        "    print(error)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "no git repository or pyproject.toml" in result.stdout


def test_override_config_restores_previous_config_on_exit() -> None:
    original_config = get_config()
    replacement_config = _Config(debug_mode=True)

    with override_config(replacement_config):
        assert get_config() is replacement_config
    assert get_config() is original_config

    with (
        pytest.raises(RuntimeError, match="boom"),
        override_config(replacement_config),
    ):
        raise RuntimeError("boom")
    assert get_config() is original_config
