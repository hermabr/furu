import pytest

import furu
from furu.config import FuruConfig


def test_get_and_set_furu_root(furu_tmp_root, tmp_path, monkeypatch) -> None:
    assert furu.get_furu_root(version_controlled=False) == furu_tmp_root / "data"
    assert furu.get_furu_root(version_controlled=True) == (
        furu_tmp_root / "furu-data" / "artifacts"
    )
    assert furu.FURU_CONFIG.raw_dir == furu_tmp_root / "raw"

    monkeypatch.delenv("FURU_SUBMITIT_PATH", raising=False)
    furu.set_furu_root(tmp_path)
    assert furu.FURU_CONFIG.base_root == tmp_path.resolve()
    assert furu.FURU_CONFIG.submitit_root == tmp_path.resolve() / "submitit"


def test_default_base_root_uses_pyproject(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname = 'demo'\n")

    monkeypatch.chdir(project_root)
    monkeypatch.delenv("FURU_PATH", raising=False)

    config = FuruConfig()

    assert config.base_root == project_root / "furu-data"


def test_default_base_root_uses_git_root(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / ".git").mkdir()

    monkeypatch.chdir(project_root)
    monkeypatch.delenv("FURU_PATH", raising=False)

    config = FuruConfig()

    assert config.base_root == project_root / "furu-data"


def test_default_base_root_falls_back_to_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FURU_PATH", raising=False)

    config = FuruConfig()

    assert config.base_root == tmp_path / "furu-data"


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


def test_version_controlled_root_uses_git_root(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / ".git").mkdir()

    monkeypatch.chdir(project_root)
    monkeypatch.delenv("FURU_VERSION_CONTROLLED_PATH", raising=False)

    config = FuruConfig()

    root = config.get_root(version_controlled=True)
    assert root == project_root / "furu-data" / "artifacts"


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


def test_retry_failed_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FURU_RETRY_FAILED", "true")
    config = FuruConfig()
    assert config.retry_failed is True


def test_retry_failed_defaults_true(monkeypatch) -> None:
    monkeypatch.delenv("FURU_RETRY_FAILED", raising=False)
    config = FuruConfig()
    assert config.retry_failed is True


def test_record_git_invalid_value_raises(monkeypatch) -> None:
    monkeypatch.setenv("FURU_RECORD_GIT", "nope")
    with pytest.raises(ValueError, match="FURU_RECORD_GIT must be one of"):
        FuruConfig()


def test_allow_no_git_origin_requires_recording(monkeypatch) -> None:
    monkeypatch.setenv("FURU_RECORD_GIT", "ignore")
    monkeypatch.setenv("FURU_ALLOW_NO_GIT_ORIGIN", "true")
    with pytest.raises(ValueError, match="FURU_ALLOW_NO_GIT_ORIGIN"):
        FuruConfig()


def test_max_wait_secs_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FURU_MAX_WAIT_SECS", "123.5")
    config = FuruConfig()
    assert config.max_wait_time_sec == 123.5
