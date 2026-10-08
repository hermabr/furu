import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ByteSize

import furu
from furu import Spec, provenance
from furu.config import _Config, _FuruProvenanceConfig
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    Provenance,
    SubmitProvenance,
)

pytestmark = pytest.mark.real_probes

EXAMPLE_PROVENANCE_JSON = """
{
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


def test_example_provenance_json_round_trips_exactly() -> None:
    prov = _example_provenance()
    assert prov.submitted.timestamp == datetime(2026, 7, 5, 14, 2, 11, tzinfo=UTC)
    assert json.loads(prov.model_dump_json()) == json.loads(EXAMPLE_PROVENANCE_JSON)


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
    identity = GitIdentity.capture(git_repo)
    assert identity.commit == _git(git_repo, "rev-parse", "HEAD")
    assert identity.branch == "main"
    assert identity.remote is None
    assert Path(identity.repo_root) == git_repo.resolve()
    assert identity.dirty is False
    assert identity.diff_stats is None


def test_git_identity_dirty_repo(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("changed\n")
    identity = GitIdentity.capture(git_repo)
    assert identity.dirty is True
    assert identity.diff_stats is not None
    assert "1 file changed" in identity.diff_stats


def test_git_identity_untracked_only_is_dirty_without_diff_stats(
    git_repo: Path,
) -> None:
    (git_repo / "new.txt").write_text("new\n")
    identity = GitIdentity.capture(git_repo)
    assert identity.dirty is True
    assert identity.diff_stats is None


def test_git_identity_detached_head(git_repo: Path) -> None:
    commit = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "checkout", "-q", commit)
    identity = GitIdentity.capture(git_repo)
    assert identity.branch is None
    assert identity.commit == commit


def test_git_identity_records_remote(git_repo: Path) -> None:
    _git(git_repo, "remote", "add", "origin", "git@example.com:t/t.git")
    identity = GitIdentity.capture(git_repo)
    assert identity.remote == "git@example.com:t/t.git"


def test_git_identity_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        GitIdentity.capture(tmp_path)


def test_capture_environment_identity_finds_project_root_from_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "pyvenv.cfg").write_text(
        "home = /x\nimplementation = CPython\nuv = 0.7.13\n"
    )
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    EnvironmentIdentity.capture.cache_clear()
    try:
        identity = EnvironmentIdentity.capture()
        assert Path(identity.project_root) == tmp_path.resolve()
        assert identity.uv == "0.7.13"
    finally:
        EnvironmentIdentity.capture.cache_clear()


_PYPROJECT = {"pyproject.toml": "[project]\n"}
_LOCKED = {**_PYPROJECT, "uv.lock": "version = 1\n"}


@pytest.mark.parametrize(
    ("files", "match"),
    [
        pytest.param({}, "no pyproject.toml", id="no-project-root"),
        pytest.param(_PYPROJECT, "uv sync", id="no-uv-lock"),
        pytest.param(_LOCKED, "not managed by uv", id="no-pyvenv-cfg"),
        pytest.param(
            {**_LOCKED, "pyvenv.cfg": "home = /x\nimplementation = CPython\n"},
            "not managed by uv",
            id="pyvenv-cfg-without-uv",
        ),
    ],
)
def test_capture_environment_identity_refuses_unmanaged_projects(
    files: dict[str, str],
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    EnvironmentIdentity.capture.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=match):
            EnvironmentIdentity.capture()
    finally:
        EnvironmentIdentity.capture.cache_clear()


def test_pytest_exemption_still_records_environment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "pyvenv.cfg").write_text("home = /x\nimplementation = CPython\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setenv("PYTEST_VERSION", pytest.__version__)
    EnvironmentIdentity.capture.cache_clear()
    try:
        identity = EnvironmentIdentity.capture()
        assert identity.uv == ""
        assert identity.uv_lock_hash.startswith("blake2s:")
        assert identity.pyproject_hash.startswith("blake2s:")
    finally:
        EnvironmentIdentity.capture.cache_clear()


def test_capture_environment_identity_is_populated() -> None:
    EnvironmentIdentity.capture.cache_clear()
    try:
        identity = EnvironmentIdentity.capture()
        assert identity.python.count(".") == 2
        assert identity.uv_lock_hash.startswith("blake2s:")
        assert identity.pyproject_hash.startswith("blake2s:")
        assert identity.furu == furu.__version__
        assert (Path(identity.project_root) / "uv.lock").is_file()
    finally:
        EnvironmentIdentity.capture.cache_clear()


def _stale_lock(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, 1, "", "The lockfile is outdated")


def _no_uv_binary(args: list[str], **kwargs: object) -> object:
    raise FileNotFoundError("uv")


@pytest.mark.parametrize(
    ("fake_run", "match"),
    [
        pytest.param(_stale_lock, "The lockfile is outdated", id="stale-lock"),
        pytest.param(_no_uv_binary, "uv executable not found", id="missing-uv-binary"),
    ],
)
def test_require_uv_raises(
    fake_run: Callable[..., object], match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    provenance._require_uv.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=match):
            provenance._require_uv()
    finally:
        provenance._require_uv.cache_clear()


class _EnforcementNode(Spec[str]):
    name: str

    def create(self) -> str:
        raise AssertionError("create() must not run when uv enforcement fails")


def test_create_fails_before_compute_without_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    EnvironmentIdentity.capture.cache_clear()
    provenance._require_uv.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="not managed by uv"):
            _EnforcementNode(name="x").create()
    finally:
        EnvironmentIdentity.capture.cache_clear()
        provenance._require_uv.cache_clear()


def test_accelerator_probe_without_nvidia_smi_falls_back_to_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_nvidia_smi(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(provenance.subprocess, "run", missing_nvidia_smi)
    try:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
        provenance._probe_accelerators.cache_clear()
        assert provenance._probe_accelerators() == ("cuda ×3",)

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        provenance._probe_accelerators.cache_clear()
        assert provenance._probe_accelerators() == ()
    finally:
        provenance._probe_accelerators.cache_clear()


def test_provenance_config_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURU_PROVENANCE__SNAPSHOT", "false")
    monkeypatch.setenv("FURU_PROVENANCE__MAX_SNAPSHOT_BYTES", "1GiB")

    config = _Config()

    assert config.provenance == _FuruProvenanceConfig(
        snapshot=False,
        max_snapshot_bytes=ByteSize(1024**3),
    )
