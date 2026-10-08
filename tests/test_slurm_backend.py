from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
import stat
import subprocess
import tarfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import furu.worker.backends.slurm.backend as slurm_backend_module
import furu.worker.backends.slurm.pool as slurm_pool_module
from furu.config import (
    _WORKER_JSON_CONFIG_FILE_ENV_VAR,
    _Config,
    _dump_worker_json_config,
    _FuruDirectories,
    _read_worker_json_config,
    get_config,
)
from furu.dag import DagNode
from furu.execution.execution_coordinator import ExecutionCoordinator, RunningJob
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.resources import Worker
from furu.snapshot import create_snapshot
from furu.testing import override_config
from furu.utils import write_private_file
from furu.worker import _cli
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.resources import (
    MemoryPerCpu,
    MemoryPerGpu,
    MemoryPerNode,
    SlurmResources,
)
from furu.worker.protocol import PoolHandoff


@pytest.fixture(autouse=True)
def _slurm_needs_snapshots(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # SlurmWorkerBackend requires snapshotting, which the test harness turns
    # off; turn it back on and stub the submit-time ``uv sync`` (the fake
    # snapshots below have no lock file to sync from).
    monkeypatch.setattr(
        slurm_backend_module,
        "subprocess",
        SimpleNamespace(run=lambda *args, **kwargs: None),
    )
    data = get_config().model_dump()
    data["provenance"]["snapshot"] = True
    with override_config(_Config.model_validate(data)):
        yield


class _StubCoordinator(ExecutionCoordinator):
    """Stands in for the ExecutionCoordinator the pool calls in-process."""

    def __init__(self, count: Callable[[int], int] | int = 0) -> None:
        super().__init__(
            max_retries_per_object=0,
            backends={},
            submit_provenance=_submit_provenance(),
        )
        self._count = count
        self.failures: list[str] = []

    def count_satisfiable_jobs(self, *, backend: object, max_workers: int) -> int:
        if isinstance(self._count, int):
            return self._count
        return self._count(max_workers)

    def fail(self, message: str) -> None:
        self.failures.append(message)


class _FakeSlurm:
    """In-process ``sbatch``/``squeue``/``sacct``/``scancel`` over one queue.

    ``queue`` holds one ``"<job id> [STATE]"`` line per job as Slurm would
    report it (the state defaults to RUNNING); ``calls`` records every argv.
    """

    def __init__(self) -> None:
        self.next_job_id = 100
        self.queue: list[str] = []
        self.calls: list[list[str]] = []

    def argvs(self, executable: str) -> list[list[str]]:
        return [argv[1:] for argv in self.calls if argv[0] == executable]

    def run(self, argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        stdout = getattr(self, argv[0])(argv[1:])
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    def _jobs(self) -> Iterator[tuple[str, str]]:
        for line in self.queue:
            job_id, _, state = line.partition(" ")
            yield job_id, state or "RUNNING"

    def sbatch(self, args: list[str]) -> str:
        job_id, self.next_job_id = self.next_job_id, self.next_job_id + 1
        arrays = [a.removeprefix("--array=0-") for a in args if a.startswith("--array")]
        tasks = range(int(arrays[-1]) + 1) if arrays else [None]
        self.queue += [f"{job_id}_{t}" if t is not None else f"{job_id}" for t in tasks]
        return f"{job_id};cluster\n"

    def squeue(self, args: list[str]) -> str:
        requested = args[args.index("--jobs") + 1].split(",")
        lines = []
        for job_id, state in self._jobs():
            allocation, _, task = job_id.partition("_")
            if allocation not in requested or state.startswith("CANCELLED"):
                continue
            if "--array" not in args or not task:
                lines.append(f"{allocation} {state}")
            elif task.startswith("["):
                # Real squeue --array expands still-queued tasks one per line.
                start, _, end = task.strip("[]").partition("-")
                for t in range(int(start), int(end or start) + 1):
                    lines.append(f"{allocation}_{t} {state}")
            else:
                lines.append(f"{job_id} {state}")
        return "".join(f"{line}\n" for line in lines)

    def sacct(self, args: list[str]) -> str:
        requested = args[args.index("-j") + 1].split(",")
        return "".join(
            f"{job_id}|{state}\n"
            for job_id, state in self._jobs()
            if job_id.partition(".")[0].partition("_")[0] in requested
        )

    def scancel(self, args: list[str]) -> str:
        self.queue = [
            line
            for line in self.queue
            if line.split()[0] not in args
            and line.split()[0].partition("_")[0] not in args
        ]
        return ""


@pytest.fixture
def fake_slurm(monkeypatch: pytest.MonkeyPatch) -> _FakeSlurm:
    slurm = _FakeSlurm()
    monkeypatch.setattr(
        slurm_pool_module,
        "subprocess",
        SimpleNamespace(run=slurm.run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    return slurm


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

    # Only the backend module's view of `threading`: other threads (the log
    # writer, for one) must keep working during the test.
    monkeypatch.setattr(
        slurm_backend_module,
        "threading",
        SimpleNamespace(Thread=NoopThread, Event=threading.Event),
    )


def _submit_provenance() -> SubmitProvenance:
    """Provenance for a pretend repo at ``/repo``, backed by an empty snapshot
    tarball so ``start_pool`` can extract it."""
    snapshot_id = "stub-snapshot"
    snapshot_dir = get_config().run_directories.snapshots / snapshot_id
    if not snapshot_dir.is_dir():
        snapshot_dir.mkdir(parents=True)
        with tarfile.open(snapshot_dir / "snapshot.tar.gz", "w:gz"):
            pass
    return SubmitProvenance(
        git=GitIdentity(
            commit="0" * 40,
            branch=None,
            remote=None,
            repo_root="/repo",
            dirty=False,
            diff_stats=None,
        ),
        environment=EnvironmentIdentity(
            python="3.12.0",
            uv="0",
            project_root="/repo",
            uv_lock_hash="blake2s:0",
            pyproject_hash="blake2s:0",
            furu="0",
        ),
        snapshot_id=snapshot_id,
        submitted=SubmitContext.capture().model_copy(update={"cwd": "/repo"}),
    )


def _code_dir(provenance: SubmitProvenance) -> Path:
    assert provenance.snapshot_id is not None
    return (
        get_config().run_directories.snapshots / provenance.snapshot_id / "code"
    ).resolve()


_WORKER_CLI_ARGS = [
    "--coordinator-file",
    "worker.config.json",
    "--pool",
    "slurm:abc",
    "--idle-timeout",
    "0.25",
    "--max-failures",
    "3",
    "--log-file",
    "worker.log",
    "--component",
    "test-worker",
    "--backend",
    "slurm",
]


def test_worker_cli_runs_worker_loop_with_parsed_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_cli, "worker_loop", lambda **kwargs: calls.append(kwargs))

    assert _cli.main(_WORKER_CLI_ARGS) == 0

    assert calls == [
        {
            "coordinator": Path("worker.config.json"),
            "pool": "slurm:abc",
            "idle_timeout": 0.25,
            "max_failures": 3,
            "component": "test-worker",
            "backend": "slurm",
            "materialize_snapshot": True,
            "log_file": Path("worker.log"),
        }
    ]


def test_worker_cli_rejects_missing_required_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli, "worker_loop", lambda **kwargs: pytest.fail("ran"))

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(_WORKER_CLI_ARGS[2:])  # no --coordinator-file

    assert exc_info.value.code == 2


def test_slurm_backend_submits_workers_with_required_sbatch_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(2)
    executor_dir = tmp_path / "furu" / "executions" / "executor-1"

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
        max_failures_per_worker=2,
        pre_worker_commands=('echo "Hello" > /tmp/hey',),
        labels=("hopper",),
    )

    provenance = _submit_provenance()
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=executor_dir,
        handoff=PoolHandoff(),
    )
    pool._scale_once()

    assert pool._job_ids == ["100_0", "100_1"]
    (worker_dir,) = (executor_dir / "workers").iterdir()
    log_dir = worker_dir / "logs"
    assert log_dir.is_dir()

    (argv,) = fake_slurm.argvs("sbatch")
    assert f"--chdir={_code_dir(provenance)}" in argv
    assert "--array=0-1" in argv
    for arg in ("--partition=debug", "--cpus-per-task=4", "--mem=8G", "--gpus=1"):
        assert arg in argv
    assert "--exclusive" in argv
    assert backend.worker == Worker(cpus=4, gpus=1, memory_gib=8, labels=("hopper",))

    script = Path(argv[-1]).read_text()
    assert script.index('echo "Hello" > /tmp/hey') < script.index("exec uv run")
    assert script.index("unset VIRTUAL_ENV") < script.index("exec uv run")
    assert f"--project {_code_dir(provenance)}" in script
    assert "--idle-timeout 0.25" in script
    assert "--max-failures 2" in script
    assert f"--pool {shlex.quote(backend.pool_key)}" in script

    # Workers find the coordinator and config through one private file; the
    # auth token never appears on a command line or in the script.
    config_file = worker_dir / "worker.config.json"
    assert f"--coordinator-file {config_file}" in script
    assert f"export {_WORKER_JSON_CONFIG_FILE_ENV_VAR}={config_file}" in script
    assert _mode(config_file) == 0o600
    coordinator_url, written_config = _read_worker_json_config(config_file)
    assert (
        coordinator_url == "ws://furu:secret-token@execution-coordinator.cluster:1234"
    )
    assert written_config == get_config()
    assert "secret-token" not in script
    assert "secret-token" not in str(fake_slurm.calls)


def test_slurm_backend_isolates_worker_files_between_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    executor_dir = tmp_path / "executor"
    coordinator = _StubCoordinator()

    pools = [
        SlurmWorkerBackend(
            max_workers=1,
            resources=SlurmResources(cpus_per_worker=1),
            pre_worker_commands=(f"echo pool-{pool_number}",),
        ).start_pool(
            coordinator=coordinator,
            bound_port=1234,
            auth_token="secret-token",
            executor_dir=executor_dir,
            handoff=PoolHandoff(),
        )
        for pool_number in (1, 2)
    ]

    worker_dirs = {pool._script_path.parent for pool in pools}
    assert len(worker_dirs) == 2
    assert {path.parent for path in worker_dirs} == {executor_dir / "workers"}
    for pool_number, pool in enumerate(pools, start=1):
        worker_dir = pool._script_path.parent
        assert f"echo pool-{pool_number}" in pool._script_path.read_text()
        coordinator_url, _ = _read_worker_json_config(worker_dir / "worker.config.json")
        assert coordinator_url.startswith("ws://furu:secret-token@")
        assert (worker_dir / "logs").is_dir()


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize(
    ("use_job_arrays", "slurm_env", "expected"),
    [
        pytest.param(False, {"SLURM_JOB_ID": "12345"}, "slurm-worker-12345", id="job"),
        pytest.param(
            True,
            {
                "SLURM_ARRAY_JOB_ID": "100",
                "SLURM_ARRAY_TASK_ID": "7",
                "SLURM_JOB_ID": "999",
            },
            "slurm-worker-100a7",
            id="array-task",
        ),
    ],
)
def test_slurm_worker_component_label_derivation_under_bash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_job_arrays: bool,
    slurm_env: dict[str, str],
    expected: str,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        use_job_arrays=use_job_arrays,
    )
    pool = backend.start_pool(
        coordinator=_StubCoordinator(),
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
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
        env={**os.environ, **slurm_env},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == expected


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
        coordinator=_StubCoordinator(),
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )

    assert (
        tuple(arg for arg in pool._sbatch_base_args if arg.startswith("--export"))
        == expected_args
    )


def test_slurm_worker_pool_ignores_untracked_array_siblings_from_sacct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(3)
    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=True,
    )
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )
    pool._scale_once()
    pool._job_ids[:] = ["100_1"]
    fake_slurm.queue = [
        "100 COMPLETED",
        "100_[0-2] COMPLETED",
        "100_0 COMPLETED",
        "100_0.batch COMPLETED",
        "100_1 RUNNING",
        "100_2 COMPLETED",
        "100_2.batch COMPLETED",
    ]

    assert pool._task_states() == {"100_1": "RUNNING"}


def test_slurm_pool_submits_replacement_workers_as_job_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(lambda max_workers: max_workers)
    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
    )
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )

    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1", "100_2"]

    fake_slurm.queue = ["100_0"]
    pool._scale_once()

    assert pool._job_ids == ["100_0", "101_0", "101_1"]
    assert [
        [arg for arg in argv if arg.startswith("--array")]
        for argv in fake_slurm.argvs("sbatch")
    ] == [["--array=0-2"], ["--array=0-1"]]


def test_slurm_pool_replaces_nonfailed_array_workers_missing_from_squeue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(lambda max_workers: max_workers)
    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=True,
    )
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )

    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1", "100_2"]

    monkeypatch.setattr(
        type(pool), "_active_job_states", lambda self: {"100_0": "RUNNING"}
    )
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

    assert pool._job_ids == ["100_0", "101_0", "101_1"]
    assert "--array=0-1" in fake_slurm.argvs("sbatch")[1]


def test_slurm_worker_pool_runs_real_slurm_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The one test that runs sbatch/squeue/sacct/scancel as real processes
    # (canned shell scripts on PATH); the rest use the in-process fake_slurm.
    _disable_slurm_pool_scale_thread(monkeypatch)
    calls = tmp_path / "calls.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        (bin_dir / name).write_text(
            "#!/bin/sh\n"
            f'echo "$(basename "$0") $*" >> {shlex.quote(str(calls))}\n'
            'case "$(basename "$0")" in\n'
            '  sbatch) echo "100;cluster" ;;\n'
            "  squeue) printf '100_0 RUNNING\\n100_1 PENDING\\n' ;;\n"
            "  sacct) printf '100_0|RUNNING\\n100_1|PENDING\\n100_1.batch|PENDING\\n' ;;\n"
            "esac\n"
        )
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    pool = SlurmWorkerBackend(
        max_workers=2,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
    ).start_pool(
        coordinator=_StubCoordinator(2),
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )

    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1"]
    assert pool._active_job_states() == {"100_0": "RUNNING", "100_1": "PENDING"}
    assert pool._task_states() == {"100_0": "RUNNING", "100_1": "PENDING"}
    pool.stop(timeout=0)

    sbatch, squeue, sacct, scancel = calls.read_text().splitlines()
    assert sbatch.split() == [
        "sbatch",
        "--parsable",
        "--array=0-1",
        *pool._sbatch_base_args,
        str(pool._script_path),
    ]
    assert squeue == "squeue --noheader --jobs 100 --array --format=%i %T"
    assert sacct == "sacct -X --noheader -o JobID,State --parsable2 -j 100"
    assert scancel == "scancel 100_0 100_1"


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


def test_slurm_backend_worker_connect_port_overrides_bound_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(1)
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        worker_connect_port=9000,
    )

    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=4321,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )
    pool._scale_once()

    (argv,) = fake_slurm.argvs("sbatch")
    script_path = Path(argv[-1])
    coordinator_url_text, _ = _read_worker_json_config(
        script_path.parent / "worker.config.json"
    )
    assert (
        coordinator_url_text
        == "ws://furu:secret-token@execution-coordinator.cluster:9000"
    )
    assert ":4321" not in script_path.read_text()


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


def test_slurm_backend_requires_snapshotting_at_construction() -> None:
    data = get_config().model_dump()
    data["provenance"]["snapshot"] = False

    with (
        override_config(_Config.model_validate(data)),
        pytest.raises(ValueError, match="snapshot = true"),
    ):
        SlurmWorkerBackend(
            max_workers=1,
            resources=SlurmResources(cpus_per_worker=1),
        )


def test_slurm_worker_pool_health_tracks_sacct_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(2)
    backend = SlurmWorkerBackend(
        max_workers=2,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )
    pool._scale_once()
    assert pool._task_states() == {"100": "RUNNING", "101": "RUNNING"}

    # Job steps (.batch, .extern) are not workers of their own.
    fake_slurm.queue = ["100", "100.batch COMPLETED", "101 FAILED", "101.extern FAILED"]

    assert pool._task_states() == {"100": "RUNNING", "101": "FAILED"}


# Each step optionally replaces what Slurm reports for the queue, sets the
# coordinator's demand, scales once, and checks the tracked job ids.
@pytest.mark.parametrize(
    ("max_workers", "steps", "sbatches", "scancels"),
    [
        pytest.param(
            3,
            [
                (None, 0, []),
                (None, 2, ["100", "101"]),
                (None, 10, ["100", "101", "102"]),
                (None, 10, ["100", "101", "102"]),
            ],
            3,
            [],
            id="grows-with-demand-up-to-max-workers",
        ),
        pytest.param(
            5,
            [(None, 1, ["100"])] * 3,
            1,
            [],
            id="does-not-resubmit-tracked-viable-job",
        ),
        pytest.param(
            3,
            [
                (None, 3, ["100", "101", "102"]),
                (["100", "101"], 3, ["100", "101", "103"]),
            ],
            4,
            [],
            id="replaces-workers-that-exit",
        ),
        pytest.param(
            3,
            [
                (None, 3, ["100", "101", "102"]),
                (["100 PENDING", "101 PENDING", "102 PENDING"], 1, ["100"]),
            ],
            3,
            [["102", "101"]],
            id="cancels-newest-queued-when-demand-drops",
        ),
        pytest.param(
            2,
            [(None, 2, ["100", "101"]), (None, 0, ["100", "101"])],
            2,
            [],
            id="never-cancels-running-workers",
        ),
        pytest.param(
            1,
            [(None, 1, ["100"]), ([], 1, ["101"]), ([], 1, ["102"])],
            3,
            [],
            id="completed-jobs-are-not-restarts",
        ),
    ],
)
def test_slurm_pool_scale_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
    max_workers: int,
    steps: list[tuple[list[str] | None, int, list[str]]],
    sbatches: int,
    scancels: list[list[str]],
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    demands = iter(demand for _, demand, _ in steps)
    backend = SlurmWorkerBackend(
        max_workers=max_workers,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        coordinator=_StubCoordinator(lambda _: next(demands)),
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )

    for queue, _, expected_job_ids in steps:
        if queue is not None:
            fake_slurm.queue = queue
        pool._scale_once()
        assert pool._job_ids == expected_job_ids

    assert fake_slurm.argvs("scancel") == scancels
    assert len(fake_slurm.argvs("sbatch")) == sbatches


@pytest.mark.parametrize("use_job_arrays", [False, True])
@pytest.mark.parametrize("max_workers", [3, 10])
def test_slurm_pool_scales_for_ready_work_while_workers_are_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
    use_job_arrays: bool,
    max_workers: int,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(2)
    backend = SlurmWorkerBackend(
        max_workers=max_workers,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=use_job_arrays,
    )
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )
    pool._scale_once()
    original_ids = list(pool._job_ids)
    assert len(original_ids) == 2
    fake_slurm.queue = [f"{job_id} RUNNING" for job_id in original_ids]
    workers = (
        ["slurm-worker-100a0", "slurm-worker-100a1"]
        if use_job_arrays
        else ["slurm-worker-100", "slurm-worker-101"]
    )
    # A batch occupies one worker; another pool's work adds no demand here.
    for i, worker in enumerate([*workers, workers[0], "slurm-worker-999"]):
        coordinator.running[str(i)] = RunningJob(
            node=cast(DagNode, object()), started_at=0, worker=worker
        )

    pool._scale_once()
    expected_total = min(4, max_workers)
    assert len(pool._job_ids) == expected_total
    added_ids = pool._job_ids[2:]
    fake_slurm.queue = [f"{job_id} RUNNING" for job_id in original_ids] + [
        f"{job_id} PENDING" for job_id in added_ids
    ]
    pool._scale_once()
    assert len(pool._job_ids) == expected_total
    assert fake_slurm.argvs("scancel") == []


def test_slurm_pool_scale_cancels_newest_queued_array_tasks_when_demand_drops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    counts = iter([2, 3, 1])
    coordinator = _StubCoordinator(lambda max_workers: next(counts))

    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=True,
    )
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )

    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1"]
    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1", "101_0"]

    # Slurm holds still-queued array tasks in aggregate bracket form; sacct
    # reports them that way while squeue --array expands them per task.
    fake_slurm.queue = ["100_0", "100_[1] PENDING", "101_[0] PENDING"]
    pool._scale_once()

    assert pool._job_ids == ["100_0"]
    assert fake_slurm.calls[-1] == ["scancel", "101_0", "100_1"]
    # The cancelled tasks are untracked now, so they can no longer make the
    # pool look unhealthy.
    assert pool._task_states() == {"100_0": "RUNNING"}


def test_slurm_pool_scale_replaces_failed_workers_within_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(lambda max_workers: max_workers)

    backend = SlurmWorkerBackend(
        max_workers=2,
        max_failed_workers=2,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )

    pool._scale_once()
    assert pool._job_ids == ["100", "101"]

    # One worker OOMs; the other exits cleanly and leaves the queue uncounted.
    fake_slurm.queue = ["100 OUT_OF_MEMORY"]
    pool._scale_once()

    assert pool._job_ids == ["102", "103"]
    assert pool._failed == ["100 OUT_OF_MEMORY"]
    assert coordinator.failures == []

    # Workers exiting non-zero after repeated job failures count as well, but a
    # job completing in the meantime is progress and clears the earlier failure.
    coordinator.completed["done"] = cast(DagNode, object())
    fake_slurm.queue = ["102 FAILED", "103 FAILED"]
    pool._scale_once()

    assert pool._job_ids == ["104", "105"]
    assert pool._failed == ["102 FAILED", "103 FAILED"]
    assert coordinator.failures == []

    # With no further progress the budget is exact: a third failure ends the run.
    fake_slurm.queue = ["104 NODE_FAIL"]
    pool._scale_once()

    assert pool._job_ids == []
    (failure,) = coordinator.failures
    assert "102 FAILED, 103 FAILED, 104 NODE_FAIL" in failure
    assert len(fake_slurm.argvs("sbatch")) == 6


def test_slurm_pool_scale_counts_cancelled_jobs_as_failed_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    coordinator = _StubCoordinator(lambda max_workers: max_workers)

    backend = SlurmWorkerBackend(
        max_workers=1,
        max_failed_workers=0,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0,
        use_job_arrays=False,
    )
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )

    pool._scale_once()
    fake_slurm.queue = ["100 CANCELLED by 12345"]
    assert pool._task_states() == {"100": "CANCELLED"}

    pool._scale_loop()

    (failure,) = coordinator.failures
    assert "100 CANCELLED" in failure
    assert pool._stop_event.is_set()
    assert pool._job_ids == []
    assert len(fake_slurm.argvs("sbatch")) == 1


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

    coordinator = _StubCoordinator()
    coordinator.submit_provenance = provenance
    pool = backend.start_pool(
        coordinator=coordinator,
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
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
    monkeypatch.setattr("furu.config._project_anchor", lambda: work_dir)
    data = get_config().model_dump()
    data["directories"] = _FuruDirectories().model_dump()  # relative furu-data/*
    backend = SlurmWorkerBackend(
        max_workers=1,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
    )

    with override_config(_Config.model_validate(data)):
        backend.start_pool(
            coordinator=_StubCoordinator(),
            bound_port=1234,
            auth_token="secret-token",
            executor_dir=tmp_path / "executor",
            handoff=PoolHandoff(),
        )

    (config_file,) = (tmp_path / "executor" / "workers").glob("*/worker.config.json")
    _, written = _read_worker_json_config(config_file)
    assert written.directories.objects == work_dir / "furu-data" / "objects"
    assert written.directories.snapshots == work_dir / "furu-data" / "snapshots"


def _slurm_backend(**overrides: Any) -> SlurmWorkerBackend:
    fields: dict[str, Any] = {
        "max_workers": 2,
        "resources": SlurmResources(partition="debug", cpus_per_worker=4),
        "worker_connect_host": "login01.cluster",
    }
    return SlurmWorkerBackend(**{**fields, **overrides})


@pytest.mark.parametrize(
    "override",
    [
        {"max_workers": 9},
        {"worker_connect_host": "login02.cluster"},
        {"worker_connect_port": 9000},
        {"job_name": "other"},
        {"poll_interval": 1.0, "worker_idle_timeout": 1.0},
    ],
    ids=lambda override: "+".join(override),
)
def test_slurm_pool_key_ignores_where_and_how_many(override: dict[str, Any]) -> None:
    key = _slurm_backend().pool_key

    assert key.startswith("slurm:")
    assert _slurm_backend(**override).pool_key == key


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
    ],
    ids=lambda override: next(iter(override)),
)
def test_slurm_pool_key_changes_with_what_a_worker_is(
    override: dict[str, Any],
) -> None:
    assert _slurm_backend(**override).pool_key != _slurm_backend().pool_key


def test_slurm_pool_key_distinguishes_memory_kinds() -> None:
    per_node = _slurm_backend(
        resources=SlurmResources(cpus_per_worker=4, memory=MemoryPerNode(8))
    )
    per_cpu = _slurm_backend(
        resources=SlurmResources(cpus_per_worker=4, memory=MemoryPerCpu(8))
    )

    assert per_node.pool_key != per_cpu.pool_key


def _wait_until(condition: Callable[[], bool], *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        time.sleep(0.01)


def test_slurm_backend_start_pool_with_handoff_inherits_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_slurm_pool_scale_thread(monkeypatch)
    inherited_dir = tmp_path / "old-executor" / "workers" / "abc"
    inherited_dir.mkdir(parents=True)
    inherited_file = inherited_dir / "worker.config.json"
    old_config = get_config().model_copy(
        update={
            "directories": get_config().directories.model_copy(
                update={
                    "objects": tmp_path / "old-objects",
                    "snapshots": tmp_path / "old-snapshots",
                }
            )
        }
    )
    write_private_file(
        inherited_file,
        _dump_worker_json_config(
            old_config, coordinator_url="ws://furu:old-token@login01:1"
        ),
        mode=0o600,
    )
    backend = SlurmWorkerBackend(
        max_workers=3,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="login02.cluster",
    )

    inherited_inode = inherited_file.stat().st_ino
    pool = backend.start_pool(
        coordinator=_StubCoordinator(),
        bound_port=4321,
        auth_token="new-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(
            job_ids=["100_0", "100_1"],
            worker_files=[inherited_file, inherited_file],
        ),
    )

    assert pool._job_ids == ["100_0", "100_1"]
    assert inherited_file.stat().st_ino == inherited_inode
    inherited_url, inherited_config = _read_worker_json_config(inherited_file)
    assert inherited_url == "ws://furu:new-token@login02.cluster:4321"
    assert inherited_config == get_config()
    assert inherited_config != old_config
    assert _mode(inherited_file) == 0o600
    (backup_file,) = inherited_dir.glob("worker.config.backup-*.json")
    backup_url, backup_config = _read_worker_json_config(backup_file)
    assert backup_url == "ws://furu:old-token@login01:1"
    assert backup_config == old_config
    assert _mode(backup_file) == 0o600
    own_file = pool._script_path.parent / "worker.config.json"
    assert own_file.read_text() == inherited_file.read_text()
    assert pool._worker_files == {own_file, inherited_file}

    assert pool.handoff() == PoolHandoff(
        job_ids=["100_0", "100_1"], worker_files=[own_file, inherited_file]
    )
    assert pool._job_ids == []

    second_backend = dataclasses.replace(backend, worker_connect_host="login03.cluster")
    second_backend.start_pool(
        coordinator=_StubCoordinator(),
        bound_port=5678,
        auth_token="newer-token",
        executor_dir=tmp_path / "second-executor",
        handoff=PoolHandoff(worker_files=[inherited_file]),
    )

    backups = list(inherited_dir.glob("worker.config.backup-*.json"))
    assert len(backups) == 2
    assert {_read_worker_json_config(path)[0] for path in backups} == {
        "ws://furu:old-token@login01:1",
        "ws://furu:new-token@login02.cluster:4321",
    }
    assert _read_worker_json_config(inherited_file)[0] == (
        "ws://furu:newer-token@login03.cluster:5678"
    )


def test_slurm_worker_pool_handoff_stops_scaling_and_stop_cancels_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_slurm: _FakeSlurm,
) -> None:
    backend = SlurmWorkerBackend(
        max_workers=2,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="execution-coordinator.cluster",
        poll_interval=0.05,
    )
    pool = backend.start_pool(
        coordinator=_StubCoordinator(2),
        bound_port=1234,
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
        handoff=PoolHandoff(),
    )
    _wait_until(lambda: len(pool._job_ids) == 2)

    handoff = pool.handoff()

    assert handoff.job_ids == ["100_0", "100_1"]
    assert not pool._scale_thread.is_alive()
    assert pool._job_ids == []

    pool.stop(timeout=0)

    assert fake_slurm.argvs("scancel") == []
    assert "100_0" in fake_slurm.queue
