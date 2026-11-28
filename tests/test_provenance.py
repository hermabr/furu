import subprocess
import sys
from pathlib import Path

import pytest
from gitrepo import git

import furu
from furu import Spec, provenance
from furu.provenance import EnvironmentIdentity, GitIdentity, SubmitContext

pytestmark = pytest.mark.usefixtures("fresh_caches")


def test_git_identity_detached_head(git_repo: Path) -> None:
    commit = git(git_repo, "rev-parse", "HEAD")
    git(git_repo, "checkout", "-q", commit)
    identity = GitIdentity.capture(git_repo)
    assert identity.branch is None
    assert identity.commit == commit


def test_git_identity_without_remote(git_repo: Path) -> None:
    git(git_repo, "remote", "remove", "origin")
    assert GitIdentity.capture(git_repo).remote is None


def test_submitting_repo_root_is_the_worktree_not_the_main_repo(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = tmp_path / "worktree"
    git(git_repo, "worktree", "add", "-q", str(worktree))
    (worktree / "sub").mkdir()
    monkeypatch.chdir(worktree / "sub")
    assert furu.submitting_repo_root() == worktree.resolve()


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
    identity = EnvironmentIdentity.capture()
    assert Path(identity.project_root) == tmp_path.resolve()
    assert identity.uv == "0.7.13"


def test_capture_environment_identity_missing_project_root_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="no pyproject.toml"):
        EnvironmentIdentity.capture()


@pytest.mark.parametrize("pyvenv_cfg", ["home = /x\nimplementation = CPython\n", None])
def test_capture_environment_identity_requires_uv_managed_python(
    pyvenv_cfg: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    if pyvenv_cfg is not None:
        (tmp_path / "pyvenv.cfg").write_text(pyvenv_cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    with pytest.raises(RuntimeError, match="not managed by uv") as excinfo:
        EnvironmentIdentity.capture()
    assert sys.executable in str(excinfo.value)
    assert str(tmp_path.resolve()) in str(excinfo.value)
    assert "uv sync" in str(excinfo.value)
    assert "uv run" in str(excinfo.value)


def test_pytest_exemption_still_records_environment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "pyvenv.cfg").write_text("home = /x\nimplementation = CPython\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setenv("PYTEST_VERSION", pytest.__version__)
    identity = EnvironmentIdentity.capture()
    assert identity.uv == ""
    assert identity.uv_lock_hash.startswith("blake2s:")
    assert identity.pyproject_hash.startswith("blake2s:")


def test_capture_environment_identity_is_cached_and_populated() -> None:
    identity = EnvironmentIdentity.capture()
    assert identity is EnvironmentIdentity.capture()
    assert identity.python.count(".") == 2
    assert identity.uv_lock_hash.startswith("blake2s:")
    assert identity.pyproject_hash.startswith("blake2s:")
    assert identity.furu == furu.__version__
    assert (Path(identity.project_root) / "uv.lock").is_file()


def test_capture_environment_identity_requires_uv_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="uv sync"):
        EnvironmentIdentity.capture()


def test_require_uv_runs_lock_check_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    provenance._require_uv()
    provenance._require_uv()
    assert calls == [["uv", "lock", "--check"]]


def test_require_uv_stale_lock_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "The lockfile is outdated")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="out of date") as excinfo:
        provenance._require_uv()
    assert "The lockfile is outdated" in str(excinfo.value)
    assert "uv sync" in str(excinfo.value)


def test_require_uv_missing_uv_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("uv")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="uv executable not found"):
        provenance._require_uv()


class _EnforcementNode(Spec):
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
    with pytest.raises(RuntimeError, match="not managed by uv"):
        _EnforcementNode(name="x").create()


def test_capture_submit_context() -> None:
    context = SubmitContext.capture()
    assert context.cwd == str(Path.cwd())
    assert context.launch_command
    assert context.timestamp.tzinfo is not None


def test_accelerator_probe_falls_back_to_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_nvidia_smi(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(provenance.subprocess, "run", missing_nvidia_smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert provenance._probe_accelerators() == ("cuda ×3",)


def test_accelerator_probe_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_nvidia_smi(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(provenance.subprocess, "run", missing_nvidia_smi)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert provenance._probe_accelerators() == ()


def test_accelerator_probe_asks_only_for_the_visible_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[list[str]] = []

    def nvidia_smi(argv: list[str], **kwargs: object) -> object:
        asked.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="NVIDIA H100\n")

    monkeypatch.setattr(provenance.subprocess, "run", nvidia_smi)
    # A Slurm step's GPUs, which the machine's other jobs do not share.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert provenance._probe_accelerators() == ("NVIDIA H100",)
    assert asked == [
        ["nvidia-smi", "--id=2", "--query-gpu=name", "--format=csv,noheader"]
    ]

    provenance._probe_accelerators.cache_clear()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert provenance._probe_accelerators() == ()
    assert len(asked) == 1  # none are visible
