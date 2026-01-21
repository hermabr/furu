import pytest

import furu
from furu.config import FuruConfig


def test_get_and_set_furu_root(furu_tmp_root, tmp_path) -> None:
    assert furu.get_furu_root(version_controlled=False) == furu_tmp_root / "data"
    assert furu.get_furu_root(version_controlled=True) == (
        furu_tmp_root / "furu-data" / "artifacts"
    )
    assert furu.FURU_CONFIG.raw_dir == furu_tmp_root / "raw"

    furu.set_furu_root(tmp_path)
    assert furu.FURU_CONFIG.base_root == tmp_path.resolve()


def test_version_controlled_root_defaults_to_pyproject(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    gitignore = project_root / ".gitignore"
    gitignore.write_text("# ignore\n")
    original_gitignore = gitignore.read_text()

    monkeypatch.chdir(project_root)
    monkeypatch.delenv("FURU_VERSION_CONTROLLED_PATH", raising=False)
    config = FuruConfig()

    root = config.get_root(version_controlled=True)
    assert root == project_root / "furu-data" / "artifacts"
    assert gitignore.read_text() == original_gitignore


def test_version_controlled_root_missing_gitignore_ok(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname = 'demo'\n")

    monkeypatch.chdir(project_root)
    monkeypatch.delenv("FURU_VERSION_CONTROLLED_PATH", raising=False)
    config = FuruConfig()

    root = config.get_root(version_controlled=True)
    assert root == project_root / "furu-data" / "artifacts"


def test_version_controlled_root_override(tmp_path, monkeypatch) -> None:
    override_root = tmp_path / "override"
    monkeypatch.setenv("FURU_VERSION_CONTROLLED_PATH", str(override_root))
    monkeypatch.chdir(tmp_path)

    config = FuruConfig()

    assert config.get_root(version_controlled=True) == override_root.resolve()


def test_always_rerun_from_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "FURU_ALWAYS_RERUN", "furu.config.FuruConfig, furu.core.furu.Furu"
    )
    config = FuruConfig()
    assert config.always_rerun == {"furu.config.FuruConfig", "furu.core.furu.Furu"}
    assert config.always_rerun_all is False


def test_always_rerun_all_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FURU_ALWAYS_RERUN", "ALL")
    config = FuruConfig()
    assert config.always_rerun_all is True
    assert config.always_rerun == set()


def test_always_rerun_all_with_entries_raises(monkeypatch) -> None:
    monkeypatch.setenv("FURU_ALWAYS_RERUN", "ALL, furu.config.FuruConfig")
    with pytest.raises(ValueError, match="cannot combine 'ALL' with specific"):
        FuruConfig()


def test_always_rerun_invalid_format_raises(monkeypatch) -> None:
    monkeypatch.setenv("FURU_ALWAYS_RERUN", "NoDot")
    with pytest.raises(ValueError, match="FURU_ALWAYS_RERUN entries must be"):
        FuruConfig()


def test_always_rerun_missing_namespace_raises(monkeypatch) -> None:
    monkeypatch.setenv("FURU_ALWAYS_RERUN", "furu.Nope")
    with pytest.raises(ValueError, match="FURU_ALWAYS_RERUN entry does not exist"):
        FuruConfig()
