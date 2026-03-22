from pathlib import Path

import furu.config as furu_config


def test_load_settings_reads_pyproject_defaults(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.furu]
debug = true
data_dir = "artifacts/furu"
""".strip(),
        encoding="utf-8",
    )
    nested_dir = tmp_path / "src" / "pkg"
    nested_dir.mkdir(parents=True)

    settings = furu_config.load_settings(start_path=nested_dir)

    assert settings.project_root == tmp_path.resolve()
    assert settings.debug is True
    assert settings.data_dir == (tmp_path / "artifacts" / "furu").resolve()


def test_load_settings_env_overrides_pyproject(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.furu]
debug = false
data_dir = "pyproject-data"
""".strip(),
        encoding="utf-8",
    )
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    monkeypatch.setenv("FURU_DEBUG", "true")
    monkeypatch.setenv("FURU_DATA_DIR", "env-data")

    settings = furu_config.load_settings(start_path=nested_dir)

    assert settings.debug is True
    assert settings.data_dir == (tmp_path / "env-data").resolve()


def test_configure_updates_settings_and_compat_config(tmp_path: Path) -> None:
    furu_config.reset_settings_for_testing(start_path=tmp_path)

    configured = furu_config.configure(data_dir="configured-data", debug=True)

    assert configured is furu_config.settings
    assert furu_config.settings.project_root == tmp_path.resolve()
    assert furu_config.settings.debug is True
    assert furu_config.settings.data_dir == (tmp_path / "configured-data").resolve()
    assert furu_config.config.debug_mode is True
    assert (
        furu_config.config.directories.data
        == (tmp_path / "configured-data").resolve()
    )
