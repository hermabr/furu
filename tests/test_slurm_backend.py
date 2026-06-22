from __future__ import annotations

import json
import os
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
    _FuruConfig,
    _FuruWorkerConfig,
    _WORKER_JSON_CONFIG_FILE_ENV_VAR,
    get_config,
)
from furu.execution.api import PoolApiClient
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
                "--resource-memory-gb",
                "0",
                "--idle-timeout",
                "60",
                "--component",
                "test-worker",
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
                "--resource-memory-gb",
                "16",
                "--idle-timeout",
                "30",
                "--component",
                "test-worker",
            ]
        )
        == 0
    )

    assert calls == [(ResourceRequest(cpus=4, gpus=1, memory_gb=16), 30.0, None)]


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
                "--resource-memory-gb",
                "0",
                "--idle-timeout",
                "0.25",
                "--component",
                "test-worker",
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
                "--resource-memory-gb",
                "0",
                "--idle-timeout",
                "0.25",
                "--component",
                "test-worker",
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
                "--resource-memory-gb",
                "0",
                "--idle-timeout",
                "60",
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
                "--resource-memory-gb",
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
                "--resource-memory-gb",
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
                "--resource-memory-gb",
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
                "--resource-memory-gb",
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
            memory=MemoryPerNode("8G"),
            memory_gb=8,
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
    )
    pool._scale_once()

    assert pool._job_ids == ["100", "101"]
    assert log_dir.is_dir()

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 2

    argv = sbatch_records[0]["argv"]
    assert "--parsable" in argv
    assert f"--chdir={work_dir.resolve()}" in argv
    assert f"--output={log_dir.resolve() / 'furu-worker-%j.out'}" in argv
    assert f"--error={log_dir.resolve() / 'furu-worker-%j.err'}" in argv
    assert "--job-name=furu-worker" in argv
    assert not any(arg.startswith("--array") for arg in argv)
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
    assert script.index('echo "Hello" > /tmp/hey') < script.index(
        f"exec {sys.executable} -m furu.worker._cli"
    )
    assert f"exec {sys.executable} -m furu.worker._cli" in script
    assert "--server-url http://execution-coordinator.cluster:1234" in script
    assert "SLURM_ARRAY_TASK_ID" not in script
    assert (
        'furu_worker_component="s${SLURM_JOB_ID:$(('
        ' ${#SLURM_JOB_ID} > 4 ? ${#SLURM_JOB_ID} - 4 : 0 ))}"' in script
    )
    assert '--component "${furu_worker_component}"' in script
    assert "--idle-timeout 0.25" in script
    assert "--max-consecutive-failures 3" in script
    assert "--resource-cpus 4" in script
    assert "--resource-gpus 1" in script
    assert "--resource-memory-gb 8" in script
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
            _FuruConfig.model_validate_json(config_file.read_text(encoding="utf-8"))
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
        ("7", "s7"),
        ("42", "s42"),
        ("999", "s999"),
        ("1000", "s1000"),
        ("12345", "s2345"),
        ("1234567", "s4567"),
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
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
    assert len(result.stdout) <= 5


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
    )
    pool._scale_once()

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 1
    assert "--export=HF_TOKEN" in sbatch_records[0]["argv"]


@pytest.mark.parametrize(
    ("memory", "expected_arg"),
    [
        (MemoryPerNode("8G"), "--mem=8G"),
        (MemoryPerCpu("2G"), "--mem-per-cpu=2G"),
        (MemoryPerGpu("16G"), "--mem-per-gpu=16G"),
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
    )
    pool._scale_once()

    assert pool._job_ids == ["100"]

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
    )
    pool._scale_once()

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 1

    script = Path(sbatch_records[0]["argv"][-1]).read_text()
    assert "--server-url http://execution-coordinator.cluster:9000" in script
    assert ":4321" not in script


def test_slurm_backend_worker_connect_host_defaults_to_config() -> None:
    config = _FuruConfig(worker=_FuruWorkerConfig(connect_host="login01.cluster"))
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
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


def test_slurm_pool_scale_does_not_count_cancelled_jobs_as_restarts(
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
    )

    pool._scale_once()
    active_file.write_text("100 CANCELLED by 12345\n")
    assert pool._task_states() == {"100": "CANCELLED"}

    pool._scale_once()

    assert pool._job_ids == ["101"]
    assert pool._failed_job_ids == []
    assert (
        len(
            [
                record
                for record in _read_records(record_file)
                if record["executable"] == "sbatch"
            ]
        )
        == 2
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
    )
    pool = backend.start_pool(
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
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

        with open(active_file, "a", encoding="utf-8") as file:
            file.write(f"{job_id}\\n")

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
                print(f"{job_id} {task_id}")
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
