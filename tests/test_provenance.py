import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ByteSize

import furu
from furu import provenance
from furu.config import _Config, _FuruProvenanceConfig
from furu.provenance import (
    EnvironmentIdentity,
    ExecuteContext,
    GitIdentity,
    NotAGitRepositoryError,
    Provenance,
    SubmitContext,
    SubmitProvenance,
    UvEnvironmentError,
    capture_environment_identity,
    capture_execute_context,
    capture_git_identity,
    capture_submit_context,
    find_project_root,
)

EXAMPLE_PROVENANCE_JSON = """
{
  "version": 1,
  "git": {
    "commit": "1fd0701e9c41d0a7b31f58c2aa04d6ce8b7712f3",
    "branch": "sweep/lr-ablation",
    "remote": "git@github.com:herman/atlas-train.git",
    "repo_root": "/home/herman/dev/atlas-train",
    "dirty": true,
    "diff_stats": "2 files changed, 31 insertions(+), 4 deletions(-)"
  },
  "environment": {
    "python": "3.12.8",
    "uv": "0.7.13",
    "project_root": "/home/herman/dev/atlas-train",
    "uv_lock_hash": "blake2s:f3ac09b1d2e44a71c8d0",
    "pyproject_hash": "blake2s:77b2c91e04d5a3f6e812",
    "furu": "0.0.62"
  },
  "snapshot_id": "9c41e2d0a7b31f58c2aa",
  "submitted": {
    "hostname": "login-01",
    "user": "herman",
    "cwd": "/home/herman/dev/atlas-train",
    "launch_command": ["uv", "run", "python", "sweep.py", "--grid", "lr"],
    "timestamp": "2026-07-05T14:02:11Z"
  },
  "executed": {
    "hostname": "gpu-node-14",
    "cpu_count": 32,
    "accelerators": ["NVIDIA H100 80GB HBM3 ×4"],
    "slurm_job_id": "48213977",
    "worker_backend": "slurm",
    "pid": 219482
  }
}
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "tracked.txt").write_text("content\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _example_provenance() -> Provenance:
    return Provenance.model_validate_json(EXAMPLE_PROVENANCE_JSON)


def test_example_provenance_json_parses() -> None:
    prov = _example_provenance()
    assert prov.version == 1
    assert prov.git is not None
    assert prov.git.dirty is True
    assert prov.submitted.launch_command == (
        "uv",
        "run",
        "python",
        "sweep.py",
        "--grid",
        "lr",
    )
    assert prov.submitted.timestamp == datetime(
        2026, 7, 5, 14, 2, 11, tzinfo=timezone.utc
    )
    assert prov.executed.worker_backend == "slurm"


def test_provenance_round_trips_through_json() -> None:
    prov = _example_provenance()
    assert Provenance.model_validate_json(prov.model_dump_json()) == prov


def test_submit_provenance_round_trips_through_json() -> None:
    prov = _example_provenance()
    submit = SubmitProvenance(
        git=prov.git,
        environment=prov.environment,
        snapshot_id=prov.snapshot_id,
        submitted=prov.submitted,
    )
    assert SubmitProvenance.model_validate_json(submit.model_dump_json()) == submit
    assert Provenance.merge(submit, prov.executed) == prov


def test_git_identity_clean_repo(git_repo: Path) -> None:
    identity = capture_git_identity(git_repo)
    assert identity.commit == _git(git_repo, "rev-parse", "HEAD")
    assert identity.branch == "main"
    assert identity.remote is None
    assert Path(identity.repo_root) == git_repo.resolve()
    assert identity.dirty is False
    assert identity.diff_stats is None


def test_git_identity_dirty_repo(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("changed\n")
    identity = capture_git_identity(git_repo)
    assert identity.dirty is True
    assert identity.diff_stats is not None
    assert "1 file changed" in identity.diff_stats


def test_git_identity_untracked_only_is_dirty_without_diff_stats(
    git_repo: Path,
) -> None:
    (git_repo / "new.txt").write_text("new\n")
    identity = capture_git_identity(git_repo)
    assert identity.dirty is True
    assert identity.diff_stats is None


def test_git_identity_detached_head(git_repo: Path) -> None:
    commit = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "checkout", "-q", commit)
    identity = capture_git_identity(git_repo)
    assert identity.branch is None
    assert identity.commit == commit


def test_git_identity_records_remote(git_repo: Path) -> None:
    _git(git_repo, "remote", "add", "origin", "git@example.com:t/t.git")
    identity = capture_git_identity(git_repo)
    assert identity.remote == "git@example.com:t/t.git"


def test_git_identity_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(NotAGitRepositoryError):
        capture_git_identity(tmp_path)


def test_uv_version_from_pyvenv_cfg(tmp_path: Path) -> None:
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_text("home = /x\nimplementation = CPython\nuv = 0.7.13\n")
    assert provenance._uv_version_from_pyvenv_cfg(cfg) == "0.7.13"

    cfg.write_text("home = /x\nimplementation = CPython\n")
    assert provenance._uv_version_from_pyvenv_cfg(cfg) is None

    assert provenance._uv_version_from_pyvenv_cfg(tmp_path / "missing.cfg") is None


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path.resolve()


def test_find_project_root_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(UvEnvironmentError, match="no pyproject.toml"):
        find_project_root(tmp_path)


def test_capture_environment_identity_is_cached_and_populated() -> None:
    capture_environment_identity.cache_clear()
    try:
        identity = capture_environment_identity()
        assert identity is capture_environment_identity()
        assert identity.python.count(".") == 2
        assert identity.uv_lock_hash.startswith("blake2s:")
        assert identity.pyproject_hash.startswith("blake2s:")
        assert identity.furu == furu.__version__
        assert (Path(identity.project_root) / "uv.lock").is_file()
    finally:
        capture_environment_identity.cache_clear()


def test_capture_environment_identity_requires_uv_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    monkeypatch.chdir(tmp_path)
    capture_environment_identity.cache_clear()
    try:
        with pytest.raises(UvEnvironmentError, match="uv sync"):
            capture_environment_identity()
    finally:
        capture_environment_identity.cache_clear()


def test_capture_submit_context() -> None:
    context = capture_submit_context()
    assert context.cwd == str(Path.cwd())
    assert context.launch_command
    assert context.timestamp.tzinfo is not None


def test_capture_execute_context_defaults_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    context = capture_execute_context()
    assert context.worker_backend == "local"
    assert context.pid == os.getpid()
    assert context.cpu_count > 0
    assert context.slurm_job_id is None
    assert context.hostname


def test_accelerator_probe_falls_back_to_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_nvidia_smi(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(provenance.subprocess, "run", missing_nvidia_smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    provenance._probe_accelerators.cache_clear()
    try:
        assert provenance._probe_accelerators() == ("cuda ×3",)
    finally:
        provenance._probe_accelerators.cache_clear()


def test_accelerator_probe_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_nvidia_smi(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(provenance.subprocess, "run", missing_nvidia_smi)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    provenance._probe_accelerators.cache_clear()
    try:
        assert provenance._probe_accelerators() == ()
    finally:
        provenance._probe_accelerators.cache_clear()


def test_provenance_config_defaults() -> None:
    config = _FuruProvenanceConfig()
    assert config.snapshot_default is False
    assert config.max_snapshot_bytes == 256 * 1024 * 1024
    assert config.verify_lock is True
    assert config.require_git == "executor"


def test_provenance_config_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURU_PROVENANCE__SNAPSHOT_DEFAULT", "true")
    monkeypatch.setenv("FURU_PROVENANCE__MAX_SNAPSHOT_BYTES", "1GiB")
    monkeypatch.setenv("FURU_PROVENANCE__VERIFY_LOCK", "false")
    monkeypatch.setenv("FURU_PROVENANCE__REQUIRE_GIT", "never")

    config = _Config()

    assert config.provenance == _FuruProvenanceConfig(
        snapshot_default=True,
        max_snapshot_bytes=ByteSize(1024**3),
        verify_lock=False,
        require_git="never",
    )


def test_provenance_config_from_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.furu.provenance]
snapshot_default = true
max_snapshot_bytes = "512MiB"
require_git = "always"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = _Config()

    assert config.provenance.snapshot_default is True
    assert config.provenance.max_snapshot_bytes == 512 * 1024 * 1024
    assert config.provenance.require_git == "always"


def test_exceptions_are_runtime_errors_and_exported() -> None:
    for exception in (
        furu.UvEnvironmentError,
        furu.SnapshotTooLargeError,
        furu.NotAGitRepositoryError,
    ):
        assert issubclass(exception, RuntimeError)


def test_example_json_matches_model_schema_exactly() -> None:
    prov = _example_provenance()
    dumped = json.loads(prov.model_dump_json())
    assert set(dumped) == set(json.loads(EXAMPLE_PROVENANCE_JSON))
    assert set(dumped["git"]) == set(GitIdentity.model_fields)
    assert set(dumped["environment"]) == set(EnvironmentIdentity.model_fields)
    assert set(dumped["submitted"]) == set(SubmitContext.model_fields)
    assert set(dumped["executed"]) == set(ExecuteContext.model_fields)
