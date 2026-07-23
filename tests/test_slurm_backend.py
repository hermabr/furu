from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import furu.worker.backends.slurm.backend as slurm_backend_module
from furu.config import (
    _Config,
    _FuruDirectories,
    _FuruWorkerConfig,
    _WORKER_JSON_CONFIG_FILE_ENV_VAR,
    get_config,
)
from furu.execution.api import PoolApiClient
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.snapshot import create_snapshot
from furu.resources import ResourceRequest
from furu.testing import override_config
from furu.worker import _cli
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.pool import _UNFINISHED_STATES
from furu.worker.backends.slurm.resources import (
    MemoryPerCpu,
    MemoryPerGpu,
    MemoryPerNode,
    SlurmResources,
)


def _stub_count_satisfiable_jobs(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    monkeypatch.setattr(
        PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: count,
    )


def _disable_slurm_pool_scale_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoopThread:
        def __init__(self, *, target: object, name: str) -> None:
            self.name = name

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

    monkeypatch.setattr(slurm_backend_module.threading, "Thread", NoopThread)


def _submit_provenance() -> SubmitProvenance:
    return SubmitProvenance(
        git=GitIdentity(
            commit="0" * 40,
            branch=None,
            remote=None,
            repo_root=".",
            dirty=False,
            diff_stats=None,
        ),
        environment=EnvironmentIdentity(
            python="3.12.0",
            uv="0",
            project_root=".",
            uv_lock_hash="blake2s:0",
            pyproject_hash="blake2s:0",
            furu="0",
        ),
        snapshot_id=None,
        submitted=SubmitContext.capture(),
    )


def test_worker_cli_reads_auth_token_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, ResourceRequest, float | None, int | None]] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret\n\n")

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
        max_consecutive_failures: int | None,
        component: str,
        backend: str,
    ) -> None:
        calls.append(
            (
                server_url,
                auth_token,
                resource_request,
                idle_timeout,
                max_consecutive_failures,
            )
        )

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    assert (
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--resource-cpus",
                "1",
                "--resource-gpus",
                "0",
                "--resource-memory-gib",
                "0",
                "--idle-timeout",
                "60",
                "--component",
                "test-worker",
                "--backend",
                "slurm",
            ]
        )
        == 0
    )

    assert calls == [
        (
            "http://execution-coordinator.test",
            "secret",
            ResourceRequest(),
            60.0,
            None,
        )
    ]
    assert token_file.exists()


def test_worker_cli_reads_resource_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[ResourceRequest, float | None, int | None]] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
        max_consecutive_failures: int | None,
        component: str,
        backend: str,
    ) -> None:
        calls.append((resource_request, idle_timeout, max_consecutive_failures))

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    assert (
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--resource-cpus",
                "4",
                "--resource-gpus",
                "1",
                "--resource-memory-gib",
                "16",
                "--idle-timeout",
                "30",
                "--component",
                "test-worker",
                "--backend",
                "slurm",
            ]
        )
        == 0
    )

    assert calls == [(ResourceRequest(cpus=4, gpus=1, memory_gib=16), 30.0, None)]


def test_worker_cli_reads_idle_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float | None] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
        max_consecutive_failures: int | None,
        component: str,
        backend: str,
    ) -> None:
        calls.append(idle_timeout)

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    assert (
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--resource-cpus",
                "4",
                "--resource-gpus",
                "1",
                "--resource-memory-gib",
                "0",
                "--idle-timeout",
                "0.25",
                "--component",
                "test-worker",
                "--backend",
                "slurm",
            ]
        )
        == 0
    )

    assert calls == [0.25]


def test_worker_cli_reads_max_consecutive_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
        max_consecutive_failures: int | None,
        component: str,
        backend: str,
    ) -> None:
        calls.append(max_consecutive_failures)

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    assert (
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--resource-cpus",
                "4",
                "--resource-gpus",
                "1",
                "--resource-memory-gib",
                "0",
                "--idle-timeout",
                "0.25",
                "--component",
                "test-worker",
                "--backend",
                "slurm",
                "--max-consecutive-failures",
                "3",
            ]
        )
        == 0
    )

    assert calls == [3]


def _run_worker_cli_capturing_component(
    monkeypatch: pytest.MonkeyPatch,
    token_file: Path,
    extra_args: list[str],
) -> str:
    captured: list[str] = []

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
        max_consecutive_failures: int | None,
        component: str,
        backend: str,
    ) -> None:
        captured.append(component)

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    assert (
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--resource-cpus",
                "1",
                "--resource-gpus",
                "0",
                "--resource-memory-gib",
                "0",
                "--idle-timeout",
                "60",
                "--backend",
                "slurm",
                *extra_args,
            ]
        )
        == 0
    )
    (component,) = captured
    return component


def test_worker_cli_reads_component_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    component = _run_worker_cli_capturing_component(
        monkeypatch, token_file, ["--component", "worker-a"]
    )

    assert component == "worker-a"


def test_worker_cli_requires_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
        max_consecutive_failures: int | None,
        component: str,
        backend: str,
    ) -> None:
        raise AssertionError("worker_loop should not be called")

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--resource-cpus",
                "1",
                "--resource-gpus",
                "0",
                "--resource-memory-gib",
                "0",
                "--idle-timeout",
                "60",
            ]
        )

    assert exc_info.value.code == 2


def test_worker_cli_requires_resource_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ResourceRequest] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
    ) -> None:
        calls.append(resource_request)

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--idle-timeout",
                "60",
                "--component",
                "test-worker",
            ]
        )

    assert exc_info.value.code == 2
    assert calls == []


def test_worker_cli_requires_auth_token_file(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
    ) -> None:
        calls.append((server_url, auth_token))

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--resource-cpus",
                "1",
                "--resource-gpus",
                "0",
                "--resource-memory-gib",
                "0",
                "--idle-timeout",
                "60",
                "--component",
                "test-worker",
            ]
        )

    assert exc_info.value.code == 2
    assert calls == []


def test_worker_cli_requires_idle_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float | None] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
    ) -> None:
        calls.append(idle_timeout)

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--resource-cpus",
                "1",
                "--resource-gpus",
                "0",
                "--resource-memory-gib",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert calls == []


def test_worker_cli_rejects_auth_token_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    def worker_loop(
        *,
        server_url: str,
        auth_token: str,
        resource_request: ResourceRequest,
        idle_timeout: float | None,
    ) -> None:
        calls.append((server_url, auth_token))

    monkeypatch.setattr(_cli, "worker_loop", worker_loop)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(
            [
                "--server-url",
                "http://execution-coordinator.test",
                "--auth-token-file",
                str(token_file),
                "--resource-cpus",
                "1",
                "--resource-gpus",
                "0",
                "--resource-memory-gib",
                "0",
                "--idle-timeout",
                "60",
                "--component",
                "test-worker",
                "--auth-token",
                "secret",
            ]
        )

    assert exc_info.value.code == 2
    assert calls == []


def test_slurm_backend_submits_workers_with_required_sbatch_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 2)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    executor_dir = tmp_path / "furu" / "executions" / "executor-1"
    worker_dir = executor_dir / "workers"
    log_dir = worker_dir / "logs"
    monkeypatch.chdir(work_dir)

    backend = SlurmWorkerBackend(
        max_workers=2,
        resources=SlurmResources(
            partition="debug",
            cpus_per_worker=4,
            memory=MemoryPerNode(8),
            gpus=1,
            extra_sbatch_args=("--exclusive",),
        ),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=1.5,
        worker_idle_timeout=0.25,
        worker_max_consecutive_failures=3,
        pre_worker_commands=('echo "Hello" > /tmp/hey',),
    )

    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=executor_dir,
        provenance=_submit_provenance(),
    )
    pool._scale_once()

    assert pool._job_ids == ["100_0", "100_1"]
    assert log_dir.is_dir()

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 1

    argv = sbatch_records[0]["argv"]
    assert "--parsable" in argv
    assert f"--chdir={work_dir.resolve()}" in argv
    assert f"--output={log_dir.resolve() / 'furu-worker-%A_%a.out'}" in argv
    assert f"--error={log_dir.resolve() / 'furu-worker-%A_%a.err'}" in argv
    assert "--job-name=furu-worker" in argv
    assert "--array=0-1" in argv
    assert not any(arg.startswith("--export") for arg in argv)
    assert not any(arg.startswith("--wrap") for arg in argv)
    assert "--partition=debug" in argv
    assert "--nodes=1" in argv
    assert "--cpus-per-task=4" in argv
    assert "--mem=8G" in argv
    assert "--gpus=1" in argv
    assert "--exclusive" in argv
    assert "secret-token" not in " ".join(argv)

    script_path = Path(argv[-1])
    script = script_path.read_text()
    assert "--auth-token-file" in script
    assert "--auth-token " not in script
    assert "secret-token" not in script
    project_root = EnvironmentIdentity.capture().project_root
    assert script.index('echo "Hello" > /tmp/hey') < script.index("exec uv run")
    assert script.index("unset VIRTUAL_ENV") < script.index("exec uv run")
    assert f"exec uv run --frozen --project {project_root}" in script
    assert "python -m furu.worker._cli" in script
    assert sys.executable not in script
    assert "--backend slurm" in script
    assert "--server-url http://execution-coordinator.cluster:1234" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "SLURM_ARRAY_JOB_ID" in script
    assert (
        'furu_worker_component="slurm-worker-${SLURM_ARRAY_JOB_ID}a${SLURM_ARRAY_TASK_ID}"'
        in script
    )
    assert '--component "${furu_worker_component}"' in script
    assert "--idle-timeout 0.25" in script
    assert "--max-consecutive-failures 3" in script
    assert "--resource-cpus 4" in script
    assert "--resource-gpus 1" in script
    assert "--resource-memory-gib 8" in script
    assert "FURU_DIRECTORIES__OBJECTS" not in script
    assert "FURU_DIRECTORIES__EXECUTIONS" not in script

    assert not (worker_dir / "secrets").exists()
    token_files = sorted(worker_dir.glob("worker-*.token"))
    assert len(token_files) == 1
    for token_file in token_files:
        assert _mode(token_file) == 0o600
        assert token_file.read_text() == "secret-token"
        assert str(token_file) in script

    config_files = sorted(worker_dir.glob("worker-*.config.json"))
    assert len(config_files) == 1
    for config_file in config_files:
        assert _mode(config_file) == 0o600
        assert (
            _Config.model_validate_json(config_file.read_text(encoding="utf-8"))
            == get_config()
        )
        assert f"export {_WORKER_JSON_CONFIG_FILE_ENV_VAR}={config_file}" in script

    assert not sbatch_records[0]["has_execution_coordinator_environment"]

    assert "secret-token" not in record_file.read_text()

    assert all(token_file.exists() for token_file in token_files)
    assert all(config_file.exists() for config_file in config_files)


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize(
    ("job_id", "expected"),
    [
        ("7", "slurm-worker-7"),
        ("42", "slurm-worker-42"),
        ("999", "slurm-worker-999"),
        ("1000", "slurm-worker-1000"),
        ("12345", "slurm-worker-12345"),
        ("1234567", "slurm-worker-1234567"),
    ],
)
def test_slurm_worker_component_label_derivation_under_bash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    job_id: str,
    expected: str,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    script_text = pool._script_path.read_text()
    component_line = next(
        line
        for line in script_text.splitlines()
        if line.startswith("furu_worker_component=")
    )
    script = (
        "set -euo pipefail\n"
        + component_line
        + "\n"
        + 'printf "%s" "$furu_worker_component"'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "SLURM_JOB_ID": job_id},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == expected


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
def test_slurm_array_worker_component_label_derivation_under_bash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        use_job_arrays=True,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    script_text = pool._script_path.read_text()
    component_line = next(
        line
        for line in script_text.splitlines()
        if line.startswith("furu_worker_component=")
    )
    script = (
        "set -euo pipefail\n"
        + component_line
        + "\n"
        + 'printf "%s" "$furu_worker_component"'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "SLURM_ARRAY_JOB_ID": "100",
            "SLURM_ARRAY_TASK_ID": "7",
            "SLURM_JOB_ID": "999",
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == "slurm-worker-100a7"


@pytest.mark.parametrize(
    ("export", "expected_args"),
    [
        (None, ()),
        ((), ()),
        ("NIL", ("--export=NIL",)),
        ("ALL", ("--export=ALL",)),
        (("HF_TOKEN", "WANDB_API_KEY"), ("--export=HF_TOKEN,WANDB_API_KEY",)),
    ],
)
def test_slurm_backend_export_option_controls_sbatch_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export: slurm_backend_module.SlurmExport,
    expected_args: tuple[str, ...],
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        export=export,
    )

    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    assert (
        tuple(arg for arg in pool._sbatch_base_args if arg.startswith("--export"))
        == expected_args
    )


def test_slurm_backend_includes_selected_export_names_in_sbatch_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 1)
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        export=("HF_TOKEN",),
    )

    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._scale_once()

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 1
    assert "--export=HF_TOKEN" in sbatch_records[0]["argv"]


def test_slurm_backend_can_submit_workers_as_job_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 3)
    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=True,
    )

    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._scale_once()

    assert pool._job_ids == ["100_0", "100_1", "100_2"]
    assert pool._active_job_ids() == {"100_0", "100_1", "100_2"}
    active_file.write_text("100_0\n100_1 FAILED\n100_2\n")
    assert pool._task_states() == {
        "100_0": "RUNNING",
        "100_1": "FAILED",
        "100_2": "RUNNING",
    }

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 1
    assert "--array=0-2" in sbatch_records[0]["argv"]
    assert any(
        arg.endswith("furu-worker-%A_%a.out") for arg in sbatch_records[0]["argv"]
    )
    assert any(
        arg.endswith("furu-worker-%A_%a.err") for arg in sbatch_records[0]["argv"]
    )
    assert "SLURM_ARRAY_TASK_ID" in Path(sbatch_records[0]["argv"][-1]).read_text()


def test_slurm_worker_pool_ignores_untracked_array_siblings_from_sacct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    _record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 3)
    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=True,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._scale_once()
    pool._job_ids[:] = ["100_1"]
    active_file.write_text(
        "100 COMPLETED\n"
        "100_[0-2] COMPLETED\n"
        "100_0 COMPLETED\n"
        "100_0.batch COMPLETED\n"
        "100_1 RUNNING\n"
        "100_2 COMPLETED\n"
        "100_2.batch COMPLETED\n"
    )

    furu_logger = logging.getLogger("furu")
    furu_logger.addHandler(caplog.handler)
    try:
        caplog.set_level(logging.WARNING, logger="furu")
        assert pool._task_states() == {"100_1": "RUNNING"}
    finally:
        furu_logger.removeHandler(caplog.handler)
    assert not any(
        "ignoring unexpected slurm job id from sacct" in record.message
        for record in caplog.records
    )


def test_slurm_pool_submits_replacement_workers_as_job_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setattr(
        PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: max_workers,
    )
    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1", "100_2"]

    active_file.write_text("100_0\n")
    pool._scale_once()

    assert pool._job_ids == ["100_0", "101_0", "101_1"]
    sbatch_records = [
        record
        for record in _read_records(record_file)
        if record["executable"] == "sbatch"
    ]
    assert [arg for arg in sbatch_records[0]["argv"] if arg.startswith("--array")] == [
        "--array=0-2"
    ]
    assert [arg for arg in sbatch_records[1]["argv"] if arg.startswith("--array")] == [
        "--array=0-1"
    ]


def test_slurm_pool_releases_nonfailed_array_workers_missing_from_squeue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    lost_workers: list[str] = []
    monkeypatch.setattr(
        PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: max_workers,
    )
    monkeypatch.setattr(
        PoolApiClient,
        "worker_lost",
        lambda self, *, worker: lost_workers.append(worker),
    )
    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=True,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1", "100_2"]

    monkeypatch.setattr(type(pool), "_active_job_ids", lambda self: {"100_0"})
    monkeypatch.setattr(
        type(pool),
        "_task_states",
        lambda self: {
            "100_0": "RUNNING",
            "100_1": "PREEMPTED",
            "100_2": "REQUEUED",
        },
    )
    pool._scale_once()

    assert lost_workers == ["slurm-worker-100a1", "slurm-worker-100a2"]
    assert pool._job_ids == ["100_0", "101_0", "101_1"]
    assert pool._failed_job_ids == []
    sbatch_records = [
        record
        for record in _read_records(record_file)
        if record["executable"] == "sbatch"
    ]
    assert [arg for arg in sbatch_records[1]["argv"] if arg.startswith("--array")] == [
        "--array=0-1"
    ]


def test_slurm_pool_releases_worker_requeued_with_same_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    lost_workers: list[str] = []
    _stub_count_satisfiable_jobs(monkeypatch, 1)
    monkeypatch.setattr(
        PoolApiClient,
        "worker_lost",
        lambda self, *, worker: lost_workers.append(worker),
    )
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    pool._scale_once()
    pool._scale_once()
    active_file.write_text("100 PENDING\n")
    pool._scale_once()
    pool._scale_once()

    assert lost_workers == ["slurm-worker-100"]
    assert pool._job_ids == ["100"]
    assert len(
        [
            record
            for record in _read_records(record_file)
            if record["executable"] == "sbatch"
        ]
    ) == 1

    active_file.write_text("100 RUNNING\n")
    pool._scale_once()
    active_file.write_text("100 REQUEUED\n")
    pool._scale_once()

    assert lost_workers == ["slurm-worker-100", "slurm-worker-100"]


def test_slurm_worker_pool_stop_cancels_array_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 2)
    backend = SlurmWorkerBackend(
        max_workers=2,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1"]

    pool.stop(timeout=0)

    assert active_file.read_text() == ""
    records = _read_records(record_file)
    assert records[-1] == {"executable": "scancel", "argv": ["100_0", "100_1"]}


@pytest.mark.parametrize(
    ("memory", "expected_arg"),
    [
        (MemoryPerNode(8), "--mem=8G"),
        (MemoryPerCpu(2), "--mem-per-cpu=2G"),
        (MemoryPerGpu(16), "--mem-per-gpu=16G"),
    ],
)
def test_slurm_resources_emit_one_memory_option(
    memory: MemoryPerNode | MemoryPerCpu | MemoryPerGpu,
    expected_arg: str,
) -> None:
    assert SlurmResources(cpus_per_worker=1, memory=memory).to_sbatch_args() == [
        "--nodes=1",
        "--cpus-per-task=1",
        expected_arg,
    ]


@pytest.mark.parametrize(
    ("resources", "expected_memory_gib"),
    [
        (SlurmResources(cpus_per_worker=4), 0),
        (SlurmResources(cpus_per_worker=4, memory=MemoryPerNode(8)), 8),
        (SlurmResources(cpus_per_worker=4, memory=MemoryPerCpu(2)), 8),
        (SlurmResources(cpus_per_worker=4, gpus=2, memory=MemoryPerGpu(16)), 32),
    ],
)
def test_slurm_resources_derive_worker_memory_gib(
    resources: SlurmResources,
    expected_memory_gib: int,
) -> None:
    assert resources.memory_gib == expected_memory_gib


@pytest.mark.parametrize(
    ("gpus", "expected_args"),
    [
        (0, []),
        (3, ["--gpus=3"]),
    ],
)
def test_slurm_resources_emit_gpu_option(gpus: int, expected_args: list[str]) -> None:
    assert SlurmResources(cpus_per_worker=1, gpus=gpus).to_sbatch_args() == [
        "--nodes=1",
        "--cpus-per-task=1",
        *expected_args,
    ]


def test_slurm_resources_emit_node_count() -> None:
    assert SlurmResources(cpus_per_worker=1, nodes=3).to_sbatch_args() == [
        "--nodes=3",
        "--cpus-per-task=1",
    ]


def test_slurm_backend_builds_server_url_from_worker_connect_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 1)
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
    )

    pool = backend.start_pool(
        bound_port=4321,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._scale_once()

    assert pool._job_ids == ["100_0"]

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 1

    script_path = Path(sbatch_records[0]["argv"][-1])
    script = script_path.read_text()
    assert "--server-url http://execution-coordinator.cluster:4321" in script
    assert f"--idle-timeout {get_config().worker.idle_timeout_seconds}" in script
    assert "--max-consecutive-failures 5" in script


def test_slurm_backend_worker_connect_port_overrides_bound_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 1)
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        worker_connect_port=9000,
    )

    pool = backend.start_pool(
        bound_port=4321,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._scale_once()

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 1

    script = Path(sbatch_records[0]["argv"][-1]).read_text()
    assert "--server-url http://execution-coordinator.cluster:9000" in script
    assert ":4321" not in script


def test_slurm_backend_worker_connect_host_defaults_to_config() -> None:
    config = _Config(worker=_FuruWorkerConfig(connect_host="login01.cluster"))
    with override_config(config):
        backend = SlurmWorkerBackend(
            max_workers=1,
            resources=SlurmResources(cpus_per_worker=1),
        )

    assert backend.worker_connect_host == "login01.cluster"


def test_slurm_backend_worker_connect_host_falls_back_to_fqdn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        slurm_backend_module.socket, "getfqdn", lambda: "node17.cluster"
    )
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
    )

    assert backend.worker_connect_host == "node17.cluster"


def test_slurm_worker_pool_health_tracks_sacct_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 2)
    backend = SlurmWorkerBackend(
        max_workers=2,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._scale_once()

    assert not any(
        state not in _UNFINISHED_STATES and state not in frozenset({"COMPLETED"})
        for state in pool._task_states().values()
    )

    active_file.write_text("100\n100.batch COMPLETED\n101 FAILED\n101.extern FAILED\n")

    states = pool._task_states()
    assert states == {"100": "RUNNING", "101": "FAILED"}
    assert any(
        state not in _UNFINISHED_STATES and state != "COMPLETED"
        for state in states.values()
    )
    sacct_records = [
        record
        for record in _read_records(record_file)
        if record["executable"] == "sacct"
    ]
    assert sacct_records[-1]["argv"] == [
        "-X",
        "--noheader",
        "-o",
        "JobID,State",
        "--parsable2",
        "-j",
        "100,101",
    ]


def test_slurm_worker_pool_unhealthy_report_includes_failed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    _record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    reports: list[str] = []
    monkeypatch.setattr(
        PoolApiClient, "fail", lambda self, *, message: reports.append(message)
    )
    backend = SlurmWorkerBackend(
        max_workers=1,
        max_failed_restarts=0,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._job_ids[:] = ["100"]
    active_file.write_text("100 FAILED\n")

    pool._scale_loop()

    assert reports == ["slurm worker pool became unhealthy: 100 FAILED"]


def test_slurm_pool_scale_submits_additional_workers_as_satisfiable_count_grows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    counts = iter([0, 2, 10, 10])
    monkeypatch.setattr(
        PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: min(next(counts), max_workers),
    )

    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    assert pool._job_ids == []

    pool._scale_once()
    assert pool._job_ids == []

    pool._scale_once()
    assert pool._job_ids == ["100", "101"]

    pool._scale_once()
    assert pool._job_ids == ["100", "101", "102"]

    pool._scale_once()
    assert pool._job_ids == ["100", "101", "102"]

    sbatch_records = [
        record
        for record in _read_records(record_file)
        if record["executable"] == "sbatch"
    ]
    assert len(sbatch_records) == 3
    assert not any(
        arg.startswith("--array") for record in sbatch_records for arg in record["argv"]
    )


def test_slurm_pool_scale_does_not_resubmit_for_already_tracked_viable_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 1)

    backend = SlurmWorkerBackend(
        max_workers=5,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    pool._scale_once()
    pool._scale_once()
    pool._scale_once()

    assert pool._job_ids == ["100"]
    sbatch_records = [
        record
        for record in _read_records(record_file)
        if record["executable"] == "sbatch"
    ]
    assert len(sbatch_records) == 1


def test_slurm_pool_scale_submits_replacement_workers_after_existing_workers_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setattr(
        PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: max_workers,
    )

    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    pool._scale_once()
    assert pool._job_ids == ["100", "101", "102"]

    active_file.write_text("100\n101\n")
    pool._scale_once()

    assert pool._job_ids == ["100", "101", "103"]
    sbatch_records = [
        record
        for record in _read_records(record_file)
        if record["executable"] == "sbatch"
    ]
    assert len(sbatch_records) == 4


def test_slurm_pool_scale_does_not_count_completed_jobs_as_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setattr(
        PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: max_workers,
    )

    backend = SlurmWorkerBackend(
        max_workers=1,
        max_failed_restarts=0,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    pool._scale_once()
    active_file.write_text("")
    pool._scale_once()
    active_file.write_text("")
    pool._scale_once()

    assert pool._job_ids == ["102"]
    assert pool._failed_job_ids == []
    assert (
        len(
            [
                record
                for record in _read_records(record_file)
                if record["executable"] == "sbatch"
            ]
        )
        == 3
    )


def test_slurm_pool_scale_reports_cancelled_jobs_as_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    reports: list[str] = []
    monkeypatch.setattr(
        PoolApiClient,
        "count_satisfiable_jobs",
        lambda self, *, resources, max_workers: max_workers,
    )
    monkeypatch.setattr(
        PoolApiClient, "fail", lambda self, *, message: reports.append(message)
    )

    backend = SlurmWorkerBackend(
        max_workers=1,
        max_failed_restarts=0,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )

    pool._scale_once()
    active_file.write_text("100 CANCELLED by 12345\n")
    assert pool._task_states() == {"100": "CANCELLED"}

    pool._scale_loop()

    assert reports == ["slurm worker pool became unhealthy: 100 CANCELLED"]
    assert pool._job_ids == ["100"]
    assert pool._failed_job_ids == ["100"]
    assert (
        len(
            [
                record
                for record in _read_records(record_file)
                if record["executable"] == "sbatch"
            ]
        )
        == 1
    )


def test_slurm_backend_requires_explicit_executor_dir() -> None:
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
    )

    with pytest.raises(TypeError, match="executor_dir"):
        backend.start_pool(
            bound_port=1234,
            auth_token="secret-token",
            provenance=_submit_provenance(),
        )  # ty: ignore[missing-argument]


def test_slurm_worker_pool_join_cancels_jobs_left_after_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    _stub_count_satisfiable_jobs(monkeypatch, 2)
    backend = SlurmWorkerBackend(
        max_workers=2,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=_submit_provenance(),
    )
    pool._scale_once()
    pool._stop_event.set()
    pool._scale_thread.start()
    pool._scale_thread.join(timeout=5)

    pool.stop(timeout=0)

    assert active_file.read_text() == ""
    records = _read_records(record_file)
    assert records[-1] == {"executable": "scancel", "argv": ["100", "101"]}


def test_slurm_backend_uses_default_poll_interval() -> None:
    with pytest.raises(TypeError):
        SlurmWorkerBackend()  # ty: ignore[missing-argument]

    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
    )

    assert backend.poll_interval == 10.0
    assert backend.execution_coordinator_listen_host == "0.0.0.0"
    assert backend.worker_idle_timeout == get_config().worker.idle_timeout_seconds
    assert backend.worker_max_consecutive_failures == 5
    assert backend.max_failed_restarts == get_config().worker.max_failed_restarts
    assert backend.use_job_arrays is True


def _install_fake_slurm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_file = tmp_path / "slurm-records.jsonl"
    active_file = tmp_path / "active-jobs.txt"
    counter_file = tmp_path / "next-job-id.txt"
    active_file.write_text("")
    counter_file.write_text("100")

    monkeypatch.delenv("FURU_EXECUTION_COORDINATOR_SERVER_URL", raising=False)
    monkeypatch.delenv("FURU_EXECUTION_COORDINATOR_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(PoolApiClient, "worker_lost", lambda self, *, worker: None)
    monkeypatch.setenv("FURU_FAKE_SLURM_RECORD_FILE", str(record_file))
    monkeypatch.setenv("FURU_FAKE_SLURM_ACTIVE_FILE", str(active_file))
    monkeypatch.setenv("FURU_FAKE_SLURM_COUNTER_FILE", str(counter_file))
    monkeypatch.setenv(
        "PATH",
        f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    )

    _write_executable(
        bin_dir / "sbatch",
        """
        import json
        import os
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]
        counter_file = os.environ["FURU_FAKE_SLURM_COUNTER_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "executable": "sbatch",
                        "argv": sys.argv[1:],
                        "has_execution_coordinator_environment": (
                            "FURU_EXECUTION_COORDINATOR_SERVER_URL" in os.environ
                            or "FURU_EXECUTION_COORDINATOR_AUTH_TOKEN" in os.environ
                        ),
                    }
                )
                + "\\n"
            )

        with open(counter_file, encoding="utf-8") as file:
            job_id = int(file.read())
        with open(counter_file, "w", encoding="utf-8") as file:
            file.write(str(job_id + 1))

        array_args = [arg for arg in sys.argv[1:] if arg.startswith("--array=0-")]
        array_end = int(array_args[-1].removeprefix("--array=0-")) if array_args else None
        task_ids = range(array_end + 1) if array_end is not None else [None]
        with open(active_file, "a", encoding="utf-8") as file:
            for task_id in task_ids:
                file.write(
                    f"{job_id}_{task_id}\\n" if task_id is not None else f"{job_id}\\n"
                )

        print(f"{job_id};cluster")
        """,
    )
    _write_executable(
        bin_dir / "sacct",
        """
        import json
        import os
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(json.dumps({"executable": "sacct", "argv": sys.argv[1:]}) + "\\n")

        requested_jobs = set()
        for index, arg in enumerate(sys.argv[1:]):
            if arg == "-j":
                requested_jobs.update(sys.argv[index + 2].split(","))
            elif arg.startswith("-j="):
                requested_jobs.update(arg.removeprefix("-j=").split(","))

        if "--noheader" not in sys.argv[1:]:
            print("JobID|State|NodeList")
        with open(active_file, encoding="utf-8") as file:
            active_jobs = file.read().splitlines()

        for active_job in sorted(active_jobs):
            job_id, _, state = active_job.partition(" ")
            allocation_job_id = job_id.partition(".")[0].partition("_")[0]
            if allocation_job_id not in requested_jobs:
                continue
            print(f"{job_id}|{state or 'RUNNING'}|node-a")
        """,
    )
    _write_executable(
        bin_dir / "squeue",
        """
        import json
        import os
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(json.dumps({"executable": "squeue", "argv": sys.argv[1:]}) + "\\n")

        requested_jobs = set()
        for index, arg in enumerate(sys.argv[1:]):
            if arg == "--jobs":
                requested_jobs.update(sys.argv[index + 2].split(","))
            elif arg.startswith("--jobs="):
                requested_jobs.update(arg.removeprefix("--jobs=").split(","))

        show_array_tasks = "--array" in sys.argv[1:]

        active_jobs = set()
        with open(active_file, encoding="utf-8") as file:
            for line in file:
                active_job, _, state = line.strip().partition(" ")
                if active_job and not state.upper().startswith("CANCELLED"):
                    active_jobs.add(active_job)

        for active_job in sorted(active_jobs):
            job_id, separator, task_id = active_job.partition("_")
            if job_id not in requested_jobs:
                continue
            if show_array_tasks and separator:
                print(f"{job_id}_{task_id}")
            else:
                print(job_id)
        """,
    )
    _write_executable(
        bin_dir / "scancel",
        """
        import json
        import os
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(json.dumps({"executable": "scancel", "argv": sys.argv[1:]}) + "\\n")

        cancelled_jobs = set(sys.argv[1:])
        with open(active_file, encoding="utf-8") as file:
            active_jobs = set(file.read().split())
        active_jobs = {
            active_job
            for active_job in active_jobs
            if active_job not in cancelled_jobs
            and active_job.partition("_")[0] not in cancelled_jobs
        }
        with open(active_file, "w", encoding="utf-8") as file:
            file.write("".join(f"{job_id}\\n" for job_id in sorted(active_jobs)))
        """,
    )

    return record_file, active_file


def _write_executable(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n{textwrap.dedent(source).lstrip()}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _read_records(record_file: Path) -> list[dict[str, Any]]:
    if not record_file.exists():
        return []
    return [json.loads(line) for line in record_file.read_text().splitlines()]


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _committed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init", "-q", "-b", "main"],
        ["add", "-A"],
        ["commit", "-qm", "init", "--allow-empty"],
    ):
        subprocess.run(
            ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    return repo


def test_slurm_backend_runs_workers_from_the_extracted_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    repo = _committed_repo(tmp_path)
    (repo / "pyproject.toml").write_text('[project]\nname = "sut"\n')
    monkeypatch.chdir(repo)
    provenance = SubmitProvenance(
        git=GitIdentity.capture(repo),
        # Hand-built: the process-wide capture is cached from the furu repo,
        # but this submit pretends to come from ``repo``.
        environment=EnvironmentIdentity(
            python="3.12.0",
            uv="0",
            project_root=str(repo),
            uv_lock_hash="blake2s:0",
            pyproject_hash="blake2s:0",
            furu="0",
        ),
        snapshot_id=create_snapshot(repo),
        submitted=SubmitContext.capture(),
    )
    uv_commands: list[list[str]] = []
    monkeypatch.setattr(
        slurm_backend_module.subprocess,
        "run",
        lambda argv, **kwargs: uv_commands.append(argv),
    )
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
    )

    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        provenance=provenance,
    )

    assert provenance.snapshot_id is not None
    code_dir = (
        get_config().run_directories.snapshots / provenance.snapshot_id / "code"
    ).resolve()
    assert (code_dir / "pyproject.toml").is_file()
    assert f"--chdir={code_dir}" in pool._sbatch_base_args
    script = pool._script_path.read_text()
    assert f"--project {shlex.quote(str(code_dir))}" in script
    assert str(repo) not in script
    # The venv is built once at submit so workers never race to create it.
    assert uv_commands == [["uv", "sync", "--frozen", "--project", str(code_dir)]]


def test_slurm_backend_pins_relative_data_directories_for_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    data = get_config().model_dump()
    data["directories"] = _FuruDirectories().model_dump()  # relative furu-data/*
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
    )

    with override_config(_Config.model_validate(data)):
        backend.start_pool(
            bound_port=1234,
            auth_token="secret-token",
            executor_dir=tmp_path / "executor",
            provenance=_submit_provenance(),
        )

    (config_file,) = (tmp_path / "executor" / "workers").glob("worker-*.config.json")
    written = _Config.model_validate_json(config_file.read_text(encoding="utf-8"))
    assert written.directories.objects == work_dir / "furu-data" / "objects"
    assert written.directories.snapshots == work_dir / "furu-data" / "snapshots"
