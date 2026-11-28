from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from gitrepo import init_repo
from websockets.sync.client import connect

import furu
import furu.execution.run_on_workers as run_module
import furu.worker.backends.slurm.pool as pool_module
from furu import plan
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
from furu.worker import _cli
from furu.worker.backends.slurm.backend import SlurmExport, SlurmWorkerBackend
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.backends.slurm.resources import (
    MemoryPerCpu,
    MemoryPerGpu,
    MemoryPerNode,
    SlurmResources,
)
from furu.worker.execute import Children
from furu.worker.messages import (
    ErrorMessage,
    Job,
    JobFailedResult,
    JobResult,
    ProcessSettings,
    SlurmPool,
    WarningMessage,
)

_URL = "ws://furu:secret-token@manager.cluster:1234"


@dataclass
class _StubManager:
    """Stands in for the Manager a pool asks for demand and reports to."""

    demand_of: Callable[[], int] | int = 0
    asked_of: list[SlurmPool] = field(
        default_factory=lambda: [_slurm_backend(max_allocations=99).pool()]
    )
    url: str = _URL
    notes: list[WarningMessage | ErrorMessage] = field(default_factory=list)

    def demand(self, key: str) -> int:
        return self.demand_of if isinstance(self.demand_of, int) else self.demand_of()

    def asked(self, key: str) -> list[SlurmPool]:
        return self.asked_of

    def notify(self, key: str, message: WarningMessage | ErrorMessage) -> None:
        self.notes.append(message)


def _slurm_backend(**overrides: Any) -> SlurmWorkerBackend:
    fields: dict[str, Any] = {
        "max_allocations": 2,
        "resources": SlurmResources(partition="debug", cpus=4),
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
            "component": "100_3",
            "backend": "slurm",
            "worker_log": Path("/logs/100_3.log"),
            "slots": 1,
            "steps": {},
        }
    ]


def test_worker_cli_reads_the_slots_and_the_step_of_each_slot_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_cli, "worker_loop", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv(_cli.MANAGER_URL_ENV_VAR, _URL)
    steps = {1: ["--exact", "--cpus-per-task=2"], 2: ["--exact", "--cpus-per-task=4"]}
    argv = [arg for item in _CLI_ARGS.items() for arg in item]

    assert _cli.main([*argv, "--slots", "2", "--steps", json.dumps(steps)]) == 0

    ((call),) = calls
    assert (call["slots"], call["steps"]) == (2, steps)


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
        resources=SlurmResources(cpus=4, memory=MemoryPerNode(8), gpus=1),
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


def test_slurm_pool_key_ignores_size_job_name_and_idle_timeout() -> None:
    key = _slurm_backend().pool_key

    assert key.startswith("slurm:")
    assert _slurm_backend(max_allocations=9).pool_key == key
    assert _slurm_backend(job_name="x", idle_timeout_seconds=5.0).pool_key == key


@pytest.mark.parametrize(
    "override",
    [
        {"resources": SlurmResources(partition="gpu", cpus=4)},
        {"resources": SlurmResources(partition="debug", cpus=8)},
        {
            "resources": SlurmResources(
                partition="debug", cpus=4, memory=MemoryPerNode(8)
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
def test_slurm_pool_key_changes_with_the_recipe(override: dict[str, Any]) -> None:
    assert _slurm_backend(**override).pool_key != _slurm_backend().pool_key


def test_slurm_pool_key_distinguishes_memory_kinds() -> None:
    per_node = _slurm_backend(resources=SlurmResources(cpus=4, memory=MemoryPerNode(8)))
    per_cpu = _slurm_backend(resources=SlurmResources(cpus=4, memory=MemoryPerCpu(8)))

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
    assert SlurmResources(cpus=1, memory=memory).to_sbatch_args() == [
        "--nodes=1",
        "--cpus-per-task=1",
        expected_arg,
    ]


@pytest.mark.parametrize(
    ("resources", "expected_memory_gib"),
    [
        (SlurmResources(cpus=4), 0),
        (SlurmResources(cpus=4, memory=MemoryPerNode(8)), 8),
        (SlurmResources(cpus=4, memory=MemoryPerCpu(2)), 8),
        (SlurmResources(cpus=4, gpus=2, memory=MemoryPerGpu(16)), 32),
    ],
)
def test_slurm_resources_derive_worker_memory_gib(
    resources: SlurmResources,
    expected_memory_gib: int,
) -> None:
    assert resources.memory_gib == expected_memory_gib


@pytest.mark.parametrize(("gpus", "expected_args"), [(0, []), (3, ["--gpus=3"])])
def test_slurm_resources_emit_gpu_option(gpus: int, expected_args: list[str]) -> None:
    assert SlurmResources(cpus=1, gpus=gpus).to_sbatch_args() == [
        "--nodes=1",
        "--cpus-per-task=1",
        *expected_args,
    ]


def test_slurm_resources_emit_node_count() -> None:
    assert SlurmResources(cpus=1, nodes=3).to_sbatch_args() == [
        "--nodes=3",
        "--cpus-per-task=1",
    ]


# --- slots ---------------------------------------------------------------------------

_NODE = SlurmResources(partition="debug", cpus=64, gpus=8, memory=MemoryPerNode(512))


def test_a_slotted_backend_offers_even_slices_fewest_slots_first() -> None:
    backend = _slurm_backend(resources=_NODE, slots=8, labels=("hopper",))

    assert backend.slices() == [
        (k, Worker(cpus=8 * k, gpus=k, memory_gib=64 * k, labels=("hopper",)))
        for k in (1, 2, 4, 8)
    ]
    assert backend.worker == backend.slices()[-1][1]
    assert _slurm_backend().slices() == [(1, _slurm_backend().worker)]


def test_slices_round_down_so_that_one_slot_each_fits() -> None:
    # 10 CPUs and 30 GiB in 3 slots: a slot has 3 CPUs and 10 GiB, all 3 the rest.
    resources = SlurmResources(cpus=10, memory=MemoryPerCpu(3))
    backend = _slurm_backend(resources=resources, slots=3)

    assert [(k, w.cpus, w.memory_gib) for k, w in backend.slices()] == [
        (1, 3, 10),
        (3, 10, 30),
    ]


def test_each_slot_count_runs_in_a_step_that_names_every_amount() -> None:
    resources = SlurmResources(cpus=64, gpus=8, memory=MemoryPerGpu(64))
    pool = _slurm_backend(resources=resources, slots=4).pool()

    step = ["--exact", "--nodes=1", "--ntasks=1"]
    assert (pool.slots, pool.steps) == (
        4,
        {
            1: [*step, "--cpus-per-task=16", "--mem=128G", "--gpus=2"],
            2: [*step, "--cpus-per-task=32", "--mem=256G", "--gpus=4"],
            4: [*step, "--cpus-per-task=64", "--mem=512G", "--gpus=8"],
        },
    )
    # The allocation itself is asked for as before.
    assert "--mem-per-gpu=64G" in pool.sbatch_args
    # One slot runs its job's child as the worker's own, with no step.
    assert _slurm_backend().pool().steps == {}


@pytest.mark.parametrize(
    ("resources", "slots", "match"),
    [
        (_NODE, 0, "slots must be at least 1"),
        (replace(_NODE, nodes=2), 2, "nodes=1, not 2"),
        (_NODE, 3, "gpus divisible by slots=3, not 8"),
        (replace(_NODE, gpus=0, cpus=4), 8, "at least slots=8 cpus, not 4"),
        (replace(_NODE, memory=None), 2, "at least slots=2 GiB of memory"),
        (replace(_NODE, memory=MemoryPerNode(4)), 8, "at least slots=8 GiB"),
    ],
)
def test_a_slotted_backend_refuses_an_allocation_it_cannot_split_evenly(
    resources: SlurmResources, slots: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _slurm_backend(resources=resources, slots=slots)


def test_slots_are_part_of_the_pool_key() -> None:
    assert (
        _slurm_backend(resources=_NODE, slots=2).pool_key
        != _slurm_backend(resources=_NODE).pool_key
    )


def test_a_slotted_pools_workers_get_its_slots_and_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path, monkeypatch, _StubManager(), resources=_NODE, slots=2)
    script = pool._script_path.read_text()

    steps = json.dumps(_slurm_backend(resources=_NODE, slots=2).pool().steps)
    assert f"    --slots 2 \\\n    --steps '{steps}' \\\n" in script
    one = _pool(tmp_path / "one", monkeypatch, _StubManager(), resources=_NODE)
    assert "--slots" not in one._script_path.read_text()


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
        get_config().directories.anchored().snapshots / provenance.git.commit
    ).resolve()
    assert code == CodeLocation(
        python=code_dir / ".venv" / "bin" / "python",
        project_root=code_dir,
        cwd=code_dir,
        repo_root=code_dir,
        checkout=repo,
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
            cpus=4,
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
    assert f'--log {worker_dir / "logs"}/"${{component}}.log"\n' in script


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
    pool = _pool(tmp_path, monkeypatch, _StubManager(3), max_allocations=3)
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
    pool = _pool(tmp_path, monkeypatch, _StubManager(3), max_allocations=3)
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
    pool = _pool(tmp_path, monkeypatch, _StubManager(3), max_allocations=3)
    pool._scale_once()
    assert pool._job_ids == ["100_0", "100_1", "100_2"]

    active_file.write_text("100_0\n100_1 COMPLETED\n100_2 COMPLETED\n")
    pool._scale_once()

    assert pool._job_ids == ["100_0", "101_0", "101_1"]
    assert [
        [arg for arg in argv if arg.startswith("--array")]
        for argv in _sbatch_argvs(record_file)
    ] == [["--array=0-2"], ["--array=0-1"]]


def test_slurm_pool_keeps_preempted_and_requeued_array_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    manager = _StubManager(3)
    pool = _pool(tmp_path, monkeypatch, manager, max_allocations=3)
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

    assert pool._job_ids == ["100_0", "100_1", "100_2"]
    assert manager.notes == []
    assert len(_sbatch_argvs(record_file)) == 1


def test_slurm_pool_keeps_every_worker_while_squeue_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    manager = _StubManager(3)
    pool = _pool(tmp_path, monkeypatch, manager, max_allocations=3)
    pool._scale_once()

    # sacct reports still-queued array tasks as one grouped row, which gives
    # the tasks no individual state; that must not look like an exit.
    active_file.write_text("100_[0-1] PENDING\n100_2 FAILED\n")

    def squeue(self: object) -> dict[str, str]:
        pool._stop_event.set()
        raise subprocess.TimeoutExpired("squeue", 60.0)

    monkeypatch.setattr(type(pool), "_active_job_states", squeue)
    pool._scale_loop()

    assert pool._job_ids == ["100_0", "100_1", "100_2"]
    assert manager.notes == []
    assert len(_sbatch_argvs(record_file)) == 1


def test_slurm_pool_health_tracks_sacct_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    pool = _pool(tmp_path, monkeypatch, _StubManager(2), use_job_arrays=False)
    pool._scale_once()

    assert pool._task_states() == {"100": "RUNNING", "101": "RUNNING"}

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
        max_allocations=3,
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


def test_slurm_pool_names_new_workers_as_the_newest_script_that_shares_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    asked = [_slurm_backend(job_name=name).pool() for name in ("a", "b")]
    manager = _StubManager(3, asked)
    pool = _pool(tmp_path, monkeypatch, manager)

    pool._scale_once()
    # No script is left that shares the pool: no worker is started.
    manager.demand_of, manager.asked_of = 5, []
    pool._scale_once()

    (argv,) = _sbatch_argvs(record_file)
    assert "--job-name=b" in argv and "--array=0-2" in argv


def test_slurm_pool_replaces_workers_that_exit_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    manager = _StubManager(3)
    pool = _pool(
        tmp_path, monkeypatch, manager, max_allocations=3, use_job_arrays=False
    )
    pool._scale_once()
    assert pool._job_ids == ["100", "101", "102"]

    active_file.write_text("100\n101\n102 COMPLETED\n")
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
        max_allocations=3,
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
    pool = _pool(
        tmp_path, monkeypatch, manager, max_allocations=1, use_job_arrays=False
    )
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


def test_slurm_pool_retries_an_sbatch_timeout_without_failing_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _StubManager(1)
    pool = _pool(tmp_path, monkeypatch, manager)

    def run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        pool._stop_event.set()
        raise subprocess.TimeoutExpired(argv, 60.0)

    monkeypatch.setattr(pool_module.subprocess, "run", run)
    pool._scale_loop()

    assert manager.notes == []
    assert pool._job_ids == []


def test_slurm_pool_keeps_scaling_after_a_scale_tick_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pool_module, "_POLL_INTERVAL_S", 0.0)
    manager = _StubManager()
    pool = _pool(tmp_path, monkeypatch, manager)
    ticks: list[int] = []

    def scale_once() -> None:
        ticks.append(len(ticks))
        if len(ticks) == 1:
            raise RuntimeError("boom")
        pool._stop_event.set()

    monkeypatch.setattr(pool, "_scale_once", scale_once)
    pool._scale_loop()

    assert ticks == [0, 1]
    assert manager.notes == [
        ErrorMessage(reason="slurm worker pool scale tick failed: RuntimeError: boom")
    ]


# --- end to end --------------------------------------------------------------------


class SlurmEndToEndLeaf(furu.Spec):
    value: int

    def create(self) -> str:
        return f"{self.value}:{os.environ['SLURM_JOB_ID']}"


def test_create_on_slurm_goes_through_the_manager_to_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setenv("FURU_FAKE_SLURM_LAUNCH", "1")
    monkeypatch.setattr(run_module, "_prepare_snapshot", lambda _: CodeLocation.here())
    backend = _slurm_backend(
        max_allocations=1,
        resources=SlurmResources(cpus=1),
        idle_timeout_seconds=30.0,
    )
    leaf = SlurmEndToEndLeaf(value=7)
    worker_dir = tmp_path / "manager" / "workers" / backend.pool_key.replace(":", "-")

    with Manager.serve(
        listen_host="127.0.0.1",
        connect_host="127.0.0.1",
        manager_dir=tmp_path / "manager",
    ) as manager:
        monkeypatch.setattr(
            run_module,
            "_connect_to_manager",
            lambda: connect(manager.url, max_size=None),
        )
        assert furu.create(leaf, on=[backend]) == "7:100"
        # A script that names its workers differently shares the pool: the
        # worker that waits takes its job.
        other = replace(backend, job_name="other", idle_timeout_seconds=0.2)
        assert furu.create(SlurmEndToEndLeaf(value=8), on=[other]) == "8:100"
        # The worker now waits as that script asked, then the manager ends it.
        log = worker_dir / "logs" / "100_0.log"
        deadline = time.monotonic() + 30
        while "manager closed the connection; exiting" not in log.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.05)

    (argv,) = _sbatch_argvs(record_file)
    assert argv[:2] == ["--parsable", "--array=0-0"]
    assert f"--chdir={worker_dir}" in argv
    assert "--job-name=furu-worker" in argv
    assert argv[-1] == str(worker_dir / "worker.sh")


class SlurmGatedLeaf(furu.Spec):
    started_file: str
    release_file: str

    def create(self) -> int:
        Path(self.started_file).touch()
        while not Path(self.release_file).exists():
            time.sleep(0.01)
        return 1


def _job(obj: furu.Spec, *, lease: int) -> Job:
    directories = get_config().directories.anchored()
    return Job(
        lease=lease,
        slots=1,
        specs=[obj.object_id],
        lines=list(map(plan.dump, plan.lines([obj]))),
        plan=directories.plans / "e1.plan",
        class_dir=obj._class_dir,
        execution_log=directories.executions / "e1" / "execution.log",
        process=ProcessSettings.from_metadata(obj._metadata),
        config=get_config().anchored(),
        code=CodeLocation.here(),
    )


class SlurmStepLeaf(furu.Spec):
    value: int
    slots: int = 1

    def runs_on(self, worker: Worker) -> bool:
        return worker.cpus >= self.slots

    def create(self) -> str:
        return (
            f"{self.value}:{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}"
        )


def test_jobs_share_a_slotted_allocation_each_in_its_own_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setenv("FURU_FAKE_SLURM_LAUNCH", "1")
    monkeypatch.setattr(run_module, "_prepare_snapshot", lambda _: CodeLocation.here())
    backend = _slurm_backend(
        max_allocations=1,
        resources=SlurmResources(cpus=2, memory=MemoryPerNode(2)),
        slots=2,
        idle_timeout_seconds=30.0,
    )
    narrow = [SlurmStepLeaf(value=value) for value in (1, 2)]
    wide = SlurmStepLeaf(value=3, slots=2)

    with Manager.serve(
        listen_host="127.0.0.1",
        connect_host="127.0.0.1",
        manager_dir=tmp_path / "manager",
    ) as manager:
        monkeypatch.setattr(
            run_module,
            "_connect_to_manager",
            lambda: connect(manager.url, max_size=None),
        )
        furu.build(narrow, on=[backend])
        # Both ran at once, each in a step of its own slot.
        ran = [leaf.load_existing().split(":") for leaf in narrow]
        assert [value for value, _ in ran] == ["1", "2"]
        assert sorted(step for _, step in ran) == ["100.0", "100.1"]
        # The job of both slots waits for their warm children to leave.
        assert furu.create(wide, on=[backend]) == "3:100.2"

    child = [sys.executable, "-m", "furu.worker._child"]
    step = ["--exact", "--nodes=1", "--ntasks=1"]
    assert [
        r["argv"] for r in _read_records(record_file) if r["executable"] == "srun"
    ] == [
        [*step, "--cpus-per-task=1", "--mem=1G", *child],
        [*step, "--cpus-per-task=1", "--mem=1G", *child],
        [*step, "--cpus-per-task=2", "--mem=2G", *child],
    ]
    (argv,) = _sbatch_argvs(record_file)
    assert "--cpus-per-task=2" in argv and "--mem=2G" in argv


def test_a_child_in_its_own_step_is_killed_with_its_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    started, release = tmp_path / "started", tmp_path / "never"
    leaf = SlurmGatedLeaf(started_file=str(started), release_file=str(release))
    steps = _slurm_backend(resources=_NODE, slots=2).pool().steps
    children = Children(worker="w", backend="slurm", slots=2, steps=steps)
    results: list[JobResult] = []
    job = threading.Thread(
        target=lambda: results.append(
            children.run(_job(leaf, lease=4), cancelled=threading.Event())
        )
    )
    job.start()
    try:
        deadline = time.monotonic() + 30
        while not started.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)

        children.kill(4)
        job.join(timeout=10)
        (result,) = results
        assert isinstance(result, JobFailedResult)
        assert result.error.endswith("subprocess died: killed by signal 9 (SIGKILL)")
        scancels = [
            r["argv"]
            for r in _read_records(record_file)
            if r["executable"] == "scancel"
        ]
        assert scancels == [["--signal=KILL", "100.0"]] * 2
        assert children._children == []
        # The folder was given back once the step was gone.
        assert not leaf.done
    finally:
        release.touch()
        children.close()
        job.join()


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
    monkeypatch.setenv("FURU_FAKE_SLURM_STEPS_FILE", str(tmp_path / "steps.txt"))
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
            print("JobID|State")
        with open(active_file, encoding="utf-8") as file:
            active_jobs = file.read().splitlines()

        for active_job in sorted(active_jobs):
            job_id, _, state = active_job.partition(" ")
            allocation_job_id = job_id.partition(".")[0].partition("_")[0]
            if allocation_job_id not in requested_jobs:
                continue
            print(f"{job_id}|{state or 'RUNNING'}")
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
    _write_executable(bin_dir / "scontrol", 'print("ClusterName             = fake")')
    _write_executable(
        bin_dir / "srun",
        """
        import fcntl
        import json
        import os
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        steps_file = os.environ["FURU_FAKE_SLURM_STEPS_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(json.dumps({"executable": "srun", "argv": sys.argv[1:]}) + "\\n")

        # A step of this job, numbered as Slurm does, that runs the command
        # in place of srun: killing the step kills this process.
        with open(steps_file, "a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            number = len(file.read().splitlines())
            file.write(f"{os.environ['SLURM_JOB_ID']}.{number} {os.getpid()}\\n")
        command = sys.argv[1:]
        while command[0].startswith("--"):
            command.pop(0)
        os.execvpe(command[0], command, dict(os.environ, SLURM_STEP_ID=str(number)))
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

        if "--signal=KILL" in sys.argv[1:]:
            import signal

            with open(os.environ["FURU_FAKE_SLURM_STEPS_FILE"], encoding="utf-8") as file:
                steps = dict(line.split() for line in file.read().splitlines())
            for step in sys.argv[1:]:
                if step in steps:
                    try:
                        os.killpg(int(steps[step]), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            sys.exit(0)

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
