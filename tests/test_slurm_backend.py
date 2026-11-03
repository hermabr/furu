from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from gitrepo import init_repo
from websockets.sync.client import connect

import furu
import furu.execution.run_on_workers as run_module
import furu.worker.backends.slurm.pool as pool_module
from furu.config import get_config
from furu.execution.manager import Manager
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.resources import Worker
from furu.snapshot import CodeLocation
from furu.testing import override_config
from furu.worker import _cli
from furu.worker.backends.slurm.backend import SlurmExport, SlurmWorkerBackend
from furu.worker.backends.slurm.pool import _UNFINISHED_STATES, SlurmWorkerPool
from furu.worker.backends.slurm.resources import (
    MemoryPerCpu,
    MemoryPerGpu,
    MemoryPerNode,
    SlurmResources,
)
from furu.worker.protocol import ErrorMessage, WarningMessage

_URL = "ws://furu:secret-token@manager.cluster:1234"


@dataclass
class _StubManager:
    """Stands in for the Manager a pool asks for demand and reports to."""

    demand_of: Callable[[], int] | int = 0
    url: str = _URL
    notes: list[WarningMessage | ErrorMessage] = field(default_factory=list)

    def demand(self, key: str) -> int:
        return self.demand_of if isinstance(self.demand_of, int) else self.demand_of()

    def notify(self, key: str, message: WarningMessage | ErrorMessage) -> None:
        self.notes.append(message)


def _slurm_backend(**overrides: Any) -> SlurmWorkerBackend:
    fields: dict[str, Any] = {
        "max_workers": 2,
        "resources": SlurmResources(partition="debug", cpus_per_worker=4),
    }
    return SlurmWorkerBackend(**{**fields, **overrides})


def _pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manager: _StubManager,
    **overrides: Any,
) -> SlurmWorkerPool:
    """A pool whose scale loop never runs on its own; tests call _scale_once."""

    class NoopThread:
        def __init__(self, *, target: object, name: str) -> None:
            pass

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

    monkeypatch.setattr(
        pool_module,
        "threading",
        SimpleNamespace(Thread=NoopThread, Event=threading.Event),
    )
    backend = _slurm_backend(**overrides)
    return SlurmWorkerPool(
        manager=cast(Any, manager),
        key=backend.pool_key,
        definition=backend.pool(),
        directory=tmp_path / "workers" / "slurm-x",
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """The pool's monotonic clock, set by assigning ``clock[0]``."""
    now = [0.0]
    monkeypatch.setattr(pool_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    yield now


def _sbatch_argvs(record_file: Path) -> list[list[str]]:
    return [
        r["argv"] for r in _read_records(record_file) if r["executable"] == "sbatch"
    ]


# --- worker CLI ------------------------------------------------------------------


_CLI_ARGS = {
    "--pool": "slurm:abc",
    "--component": "100_3",
    "--log": "/logs/100_3.log",
    "--idle-timeout": "0.25",
}


def test_worker_cli_passes_arguments_and_reads_the_url_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_cli, "worker_loop", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv(_cli.MANAGER_URL_ENV_VAR, _URL)

    assert _cli.main([arg for item in _CLI_ARGS.items() for arg in item]) == 0

    assert calls == [
        {
            "url": _URL,
            "pool": "slurm:abc",
            "idle_timeout": 0.25,
            "component": "100_3",
            "backend": "slurm",
            "worker_log": Path("/logs/100_3.log"),
        }
    ]


@pytest.mark.parametrize("missing", list(_CLI_ARGS))
def test_worker_cli_requires_every_argument(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(_cli, "worker_loop", lambda **kwargs: None)
    argv = [arg for item in _CLI_ARGS.items() if item[0] != missing for arg in item]
    with pytest.raises(SystemExit):
        _cli.main(argv)


# --- backend -----------------------------------------------------------------------


def test_slurm_backend_describes_its_worker() -> None:
    backend = _slurm_backend(
        resources=SlurmResources(cpus_per_worker=4, memory=MemoryPerNode(8), gpus=1),
        labels=("hopper",),
    )
    assert backend.worker == Worker(cpus=4, gpus=1, memory_gib=8, labels=("hopper",))


@pytest.mark.parametrize(
    ("export", "expected_args"),
    [
        (None, []),
        ((), []),
        ("NIL", ["--export=NIL"]),
        ("ALL", ["--export=ALL"]),
        (("HF_TOKEN", "WANDB_API_KEY"), ["--export=HF_TOKEN,WANDB_API_KEY"]),
    ],
)
def test_slurm_backend_export_option_controls_sbatch_args(
    export: SlurmExport, expected_args: list[str]
) -> None:
    sbatch_args = _slurm_backend(export=export).pool().sbatch_args
    assert [arg for arg in sbatch_args if arg.startswith("--export")] == expected_args


def test_slurm_pool_key_ignores_how_many() -> None:
    key = _slurm_backend().pool_key

    assert key.startswith("slurm:")
    assert _slurm_backend(max_workers=9).pool_key == key


@pytest.mark.parametrize(
    "override",
    [
        {"resources": SlurmResources(partition="gpu", cpus_per_worker=4)},
        {"resources": SlurmResources(partition="debug", cpus_per_worker=8)},
        {
            "resources": SlurmResources(
                partition="debug", cpus_per_worker=4, memory=MemoryPerNode(8)
            )
        },
        {"labels": ("hopper",)},
        {"pre_worker_commands": ("module load cuda",)},
        {"export": "NIL"},
        {"export": ("HF_TOKEN",)},
        {"use_job_arrays": False},
        {"job_name": "other"},
    ],
    ids=lambda override: next(iter(override)),
)
def test_slurm_pool_key_changes_with_the_recipe(override: dict[str, Any]) -> None:
    assert _slurm_backend(**override).pool_key != _slurm_backend().pool_key


def test_slurm_pool_key_distinguishes_memory_kinds() -> None:
    per_node = _slurm_backend(
        resources=SlurmResources(cpus_per_worker=4, memory=MemoryPerNode(8))
    )
    per_cpu = _slurm_backend(
        resources=SlurmResources(cpus_per_worker=4, memory=MemoryPerCpu(8))
    )

    assert per_node.pool_key != per_cpu.pool_key


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


@pytest.mark.parametrize(("gpus", "expected_args"), [(0, []), (3, ["--gpus=3"])])
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


def test_prepare_snapshot_builds_the_extracted_commits_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "sut"\n')
    init_repo(repo)
    monkeypatch.chdir(repo)
    provenance = SubmitProvenance(
        git=GitIdentity.capture(repo),
        # Hand-built: the process-wide capture is cached from the furu repo,
        # but this submit pretends to come from ``repo``.
        environment=EnvironmentIdentity.capture().model_copy(
            update={"project_root": str(repo)}
        ),
        submitted=SubmitContext.capture(),
    )
    uv_commands: list[list[str]] = []
    monkeypatch.setattr(
        run_module,
        "subprocess",
        SimpleNamespace(run=lambda argv, **kwargs: uv_commands.append(argv)),
    )

    code = run_module._prepare_snapshot(provenance)

    code_dir = (
        get_config().run_directories.snapshots / provenance.git.commit
    ).resolve()
    assert code == CodeLocation(
        python=code_dir / ".venv" / "bin" / "python",
        project_root=code_dir,
        cwd=code_dir,
        repo_root=code_dir,
    )
    assert (code_dir / "pyproject.toml").is_file()
    # The venv is built once at submit so workers never race to create it.
    assert uv_commands == [["uv", "sync", "--frozen", "--project", str(code_dir)]]


# --- pool --------------------------------------------------------------------------


def test_slurm_pool_submits_workers_that_run_the_managers_furu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    pool = _pool(
        tmp_path,
        monkeypatch,
        _StubManager(2),
        resources=SlurmResources(
            partition="debug",
            cpus_per_worker=4,
            memory=MemoryPerNode(8),
            gpus=1,
            extra_sbatch_args=("--exclusive",),
        ),
        pre_worker_commands=('echo "Hello" > /tmp/hey',),
    )
    pool._scale_once()

    assert pool._job_ids == ["100_0", "100_1"]
    worker_dir = tmp_path / "workers" / "slurm-x"
    (argv,) = _sbatch_argvs(record_file)
    assert argv[:2] == ["--parsable", "--array=0-1"]
    assert f"--chdir={worker_dir}" in argv
    # One file per worker, named like the worker: stdout and stderr merged.
    assert f"--output={worker_dir / 'logs' / '%A_%a.log'}" in argv
    assert not any(arg.startswith(("--error", "--export", "--wrap")) for arg in argv)
    for arg in (
        "--job-name=furu-worker",
        "--partition=debug",
        "--nodes=1",
        "--cpus-per-task=4",
        "--mem=8G",
        "--gpus=1",
        "--exclusive",
    ):
        assert arg in argv
    assert "secret-token" not in record_file.read_text()

    script_path = Path(argv[-1])
    assert script_path == worker_dir / "worker.sh"
    assert stat.S_IMODE(script_path.stat().st_mode) == 0o700
    script = script_path.read_text()
    exec_at = script.index(f"exec {sys.executable} -m furu.worker._cli")
    assert script.index('echo "Hello" > /tmp/hey') < exec_at
    assert script.index("unset VIRTUAL_ENV") < exec_at
    assert f"export {_cli.MANAGER_URL_ENV_VAR}={_URL}" in script
    assert script.count("secret-token") == 1  # in the environment, not argv
    assert f"--pool {pool._key}" in script
    assert '--component "${component}"' in script
    assert f'--log {worker_dir / "logs"}/"${{component}}.log"' in script
    assert f"--idle-timeout {get_config().worker.idle_timeout_seconds}" in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize(
    ("use_job_arrays", "environment", "expected"),
    [
        (False, {"SLURM_JOB_ID": "42"}, "42"),
        (
            True,
            {
                "SLURM_ARRAY_JOB_ID": "100",
                "SLURM_ARRAY_TASK_ID": "7",
                "SLURM_JOB_ID": "9",
            },
            "100_7",
        ),
    ],
)
def test_slurm_worker_is_named_after_its_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_job_arrays: bool,
    environment: dict[str, str],
    expected: str,
) -> None:
    pool = _pool(tmp_path, monkeypatch, _StubManager(), use_job_arrays=use_job_arrays)
    (component_line,) = [
        line
        for line in pool._script_path.read_text().splitlines()
        if line.startswith("component=")
    ]
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail\n{component_line}\nprintf "%s" "$component"',
        ],
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == expected


def test_slurm_pool_tracks_job_array_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    pool = _pool(tmp_path, monkeypatch, _StubManager(3), max_workers=3)
    pool._scale_once()

    assert pool._job_ids == ["100_0", "100_1", "100_2"]
    assert pool._active_job_states() == {
        "100_0": "RUNNING",
        "100_1": "RUNNING",
        "100_2": "RUNNING",
    }
    active_file.write_text("100_0\n100_1 FAILED\n100_2\n")
    assert pool._task_states() == {
        "100_0": "RUNNING",
        "100_1": "FAILED",
        "100_2": "RUNNING",
    }


def test_slurm_pool_ignores_untracked_array_siblings_from_sacct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    pool = _pool(tmp_path, monkeypatch, _StubManager(3), max_workers=3)
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

    furu_logger = furu.get_logger()
    furu_logger.addHandler(caplog.handler)
    try:
        assert pool._task_states() == {"100_1": "RUNNING"}
    finally:
        furu_logger.removeHandler(caplog.handler)
    assert not any("unexpected slurm job id" in message for message in caplog.messages)


def test_slurm_pool_replaces_vanished_array_workers_as_one_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    pool = _pool(tmp_path, monkeypatch, _StubManager(3), max_workers=3)
    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1", "100_2"]

    active_file.write_text("100_0\n")
    pool._scale_once()

    assert pool._job_ids == ["100_0", "101_0", "101_1"]
    assert [
        [arg for arg in argv if arg.startswith("--array")]
        for argv in _sbatch_argvs(record_file)
    ] == [["--array=0-2"], ["--array=0-1"]]


def test_slurm_pool_replaces_requeued_array_workers_missing_from_squeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    manager = _StubManager(3)
    pool = _pool(tmp_path, monkeypatch, manager, max_workers=3)
    pool._scale_once()
    monkeypatch.setattr(
        type(pool), "_active_job_states", lambda self: {"100_0": "RUNNING"}
    )
    monkeypatch.setattr(
        type(pool),
        "_task_states",
        lambda self: {"100_0": "RUNNING", "100_1": "PREEMPTED", "100_2": "REQUEUED"},
    )
    pool._scale_once()

    assert pool._job_ids == ["100_0", "101_0", "101_1"]
    assert manager.notes == []  # preempted is not failed
    assert "--array=0-1" in _sbatch_argvs(record_file)[1]


def test_slurm_pool_health_tracks_sacct_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    pool = _pool(tmp_path, monkeypatch, _StubManager(2), use_job_arrays=False)
    pool._scale_once()

    assert set(pool._task_states().values()) <= _UNFINISHED_STATES

    active_file.write_text("100\n100.batch COMPLETED\n101 FAILED\n101.extern FAILED\n")

    assert pool._task_states() == {"100": "RUNNING", "101": "FAILED"}
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


def test_slurm_pool_follows_demand_up_to_what_it_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    demands = iter([0, 2, 3, 3])
    pool = _pool(
        tmp_path,
        monkeypatch,
        _StubManager(lambda: next(demands)),
        max_workers=3,
        use_job_arrays=False,
    )

    pool._scale_once()
    assert pool._job_ids == []
    pool._scale_once()
    assert pool._job_ids == ["100", "101"]
    pool._scale_once()
    assert pool._job_ids == ["100", "101", "102"]
    pool._scale_once()
    assert pool._job_ids == ["100", "101", "102"]
    assert len(_sbatch_argvs(record_file)) == 3


def test_slurm_pool_replaces_workers_that_exit_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    manager = _StubManager(3)
    pool = _pool(tmp_path, monkeypatch, manager, max_workers=3, use_job_arrays=False)
    pool._scale_once()
    assert pool._job_ids == ["100", "101", "102"]

    active_file.write_text("100\n101\n")  # 102 finished and left sacct
    pool._scale_once()

    assert pool._job_ids == ["100", "101", "103"]
    assert len(_sbatch_argvs(record_file)) == 4
    assert manager.notes == []


@pytest.mark.parametrize("use_job_arrays", [False, True])
def test_slurm_pool_cancels_newest_queued_workers_when_demand_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_job_arrays: bool
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    demands = iter([2, 3, 1])
    pool = _pool(
        tmp_path,
        monkeypatch,
        _StubManager(lambda: next(demands)),
        max_workers=3,
        use_job_arrays=use_job_arrays,
    )
    pool._scale_once()
    pool._scale_once()
    if use_job_arrays:
        assert pool._job_ids == ["100_0", "100_1", "101_0"]
        # Slurm holds still-queued array tasks in aggregate bracket form; sacct
        # reports them that way while squeue --array expands them per task.
        active_file.write_text("100_0\n100_[1] PENDING\n101_[0] PENDING\n")
    else:
        assert pool._job_ids == ["100", "101", "102"]
        active_file.write_text("100 RUNNING\n101 PENDING\n102 PENDING\n")

    pool._scale_once()

    first, second, third = (
        ["100_0", "100_1", "101_0"] if use_job_arrays else ["100", "101", "102"]
    )
    assert pool._job_ids == [first]
    assert _read_records(record_file)[-1] == {
        "executable": "scancel",
        "argv": [third, second],
    }


def test_slurm_pool_never_cancels_running_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    demands = iter([2, 0])
    pool = _pool(
        tmp_path,
        monkeypatch,
        _StubManager(lambda: next(demands)),
        use_job_arrays=False,
    )
    pool._scale_once()
    pool._scale_once()

    assert pool._job_ids == ["100", "101"]
    assert not any(r["executable"] == "scancel" for r in _read_records(record_file))


def test_slurm_pool_stop_cancels_only_queued_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    pool = _pool(tmp_path, monkeypatch, _StubManager(2), use_job_arrays=False)
    pool._scale_once()
    active_file.write_text("100 RUNNING\n101 PENDING\n")

    pool.stop()

    assert _read_records(record_file)[-1] == {"executable": "scancel", "argv": ["101"]}
    assert active_file.read_text() == "100 RUNNING\n"


def test_failing_slurm_pool_warns_and_replaces_one_worker_per_minute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    manager = _StubManager(2)
    pool = _pool(tmp_path, monkeypatch, manager, use_job_arrays=False)
    pool._scale_once()

    clock[0] = 10.0
    active_file.write_text("100 OUT_OF_MEMORY\n101 CANCELLED by 12345\n")
    pool._scale_once()

    log_dir = tmp_path / "workers" / "slurm-x" / "logs"
    assert manager.notes == [
        WarningMessage(
            reason=f"slurm worker 100 OUT_OF_MEMORY → {log_dir / '100.log'}"
        ),
        WarningMessage(reason=f"slurm worker 101 CANCELLED → {log_dir / '101.log'}"),
    ]
    assert pool._job_ids == ["102"]  # the first replacement goes at once

    active_file.write_text("102\n")
    clock[0] = 20.0
    pool._scale_once()
    assert pool._job_ids == ["102"]  # the next waits a minute
    clock[0] = 70.0
    pool._scale_once()
    assert pool._job_ids == ["102", "103"]

    active_file.write_text("102\n103 FAILED\n")
    clock[0] = 80.0
    pool._scale_once()
    assert pool._job_ids == ["102"]
    # A completed job ends the streak; the pool fills up at once.
    pool.job_completed()
    pool._scale_once()
    assert pool._job_ids == ["102", "104"]
    assert len(_sbatch_argvs(record_file)) == 5


def test_slurm_pool_is_broken_after_many_failures_over_a_long_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    monkeypatch.setattr(pool_module, "_BROKEN_AFTER_FAILURES", 1)
    monkeypatch.setattr(pool_module, "_BROKEN_AFTER_S", 100.0)
    _record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    manager = _StubManager(1)
    pool = _pool(tmp_path, monkeypatch, manager, max_workers=1, use_job_arrays=False)
    pool._scale_once()

    for failed, now in (("100", 10.0), ("101", 80.0), ("102", 150.0)):
        clock[0] = now
        active_file.write_text(f"{failed} FAILED\n")
        pool._scale_once()  # each failure is replaced: a minute has passed

    # Two failures in 70 s are not enough; the third, 140 s in, is.
    assert [type(note) for note in manager.notes] == [
        WarningMessage,
        WarningMessage,
        WarningMessage,
        ErrorMessage,
    ]
    assert manager.notes[-1].reason == (
        "slurm pool is broken: 3 workers failed over 2m20s without a job "
        "completing: 100 FAILED, 101 FAILED, 102 FAILED"
    )
    assert pool._streak == []


# --- end to end --------------------------------------------------------------------


class SlurmEndToEndLeaf(furu.Spec[str]):
    value: int

    def create(self) -> str:
        return f"{self.value}:{os.environ['SLURM_JOB_ID']}"


def test_create_on_slurm_goes_through_the_manager_to_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setenv("FURU_FAKE_SLURM_LAUNCH", "1")
    monkeypatch.setattr(run_module, "_prepare_snapshot", lambda _: CodeLocation.here())
    config = get_config()
    quick_idle = config.model_copy(
        update={
            "worker": config.worker.model_copy(update={"idle_timeout_seconds": 5.0})
        }
    )
    backend = _slurm_backend(max_workers=1, resources=SlurmResources(cpus_per_worker=1))
    leaf = SlurmEndToEndLeaf(value=7)

    with (
        override_config(quick_idle),
        Manager.serve(
            listen_host="127.0.0.1",
            connect_host="127.0.0.1",
            manager_dir=tmp_path / "manager",
        ) as manager,
    ):
        monkeypatch.setattr(
            run_module,
            "_connect_to_manager",
            lambda: connect(manager.url, max_size=None),
        )
        assert furu.create(leaf, on=[backend]) == "7:100"

    worker_dir = tmp_path / "manager" / "workers" / backend.pool_key.replace(":", "-")
    (argv,) = _sbatch_argvs(record_file)
    assert argv[:2] == ["--parsable", "--array=0-0"]
    assert f"--chdir={worker_dir}" in argv
    assert "--job-name=furu-worker" in argv
    assert argv[-1] == str(worker_dir / "worker.sh")


# --- fake Slurm ----------------------------------------------------------------------


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

        if os.environ.get("FURU_FAKE_SLURM_LAUNCH"):
            # Start each worker for real, as Slurm would on a compute node.
            import subprocess

            script = sys.argv[-1]
            for task_id in task_ids:
                env = dict(os.environ, SLURM_JOB_ID=str(job_id))
                name = str(job_id)
                if task_id is not None:
                    env.update(
                        SLURM_ARRAY_JOB_ID=str(job_id), SLURM_ARRAY_TASK_ID=str(task_id)
                    )
                    name = f"{job_id}_{task_id}"
                log = os.path.join(os.path.dirname(script), "logs", f"{name}.log")
                subprocess.Popen(
                    ["bash", script],
                    env=env,
                    stdout=open(log, "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
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

        active_jobs = {}
        with open(active_file, encoding="utf-8") as file:
            for line in file:
                active_job, _, state = line.strip().partition(" ")
                if active_job and not state.upper().startswith("CANCELLED"):
                    active_jobs[active_job] = state or "RUNNING"

        for active_job, state in sorted(active_jobs.items()):
            job_id, separator, task_spec = active_job.partition("_")
            if job_id not in requested_jobs:
                continue
            if not (show_array_tasks and separator):
                print(f"{job_id} {state}")
            elif task_spec.startswith("["):
                # Real squeue --array expands still-queued tasks one per line.
                for part in task_spec.strip("[]").partition("%")[0].split(","):
                    start, dash, end = part.partition("-")
                    task_ids = range(int(start), int(end) + 1) if dash else [int(start)]
                    for task_id in task_ids:
                        print(f"{job_id}_{task_id} {state}")
            else:
                print(f"{job_id}_{task_spec} {state}")
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
            active_jobs = [
                line
                for line in file.read().splitlines()
                if line.split(maxsplit=1)[0] not in cancelled_jobs
                and line.split(maxsplit=1)[0].partition("_")[0] not in cancelled_jobs
            ]
        with open(active_file, "w", encoding="utf-8") as file:
            file.write("".join(f"{line}\\n" for line in sorted(active_jobs)))
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
