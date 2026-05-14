from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from furu.worker.backends.slurm import (
    SlurmResources,
    SlurmWorkerBackend,
    SlurmWorkerPool,
    _rewrite_host,
)


def test_slurm_resources_to_sbatch_args_omits_none_fields() -> None:
    assert SlurmResources().to_sbatch_args() == []


def test_slurm_resources_to_sbatch_args_full() -> None:
    resources = SlurmResources(
        partition="gpu",
        account="my-account",
        time="01:00:00",
        cpus_per_task=4,
        mem="16G",
        gres="gpu:1",
        nodes=1,
        qos="normal",
        constraint="a100",
        extra_sbatch_args=("--mail-type", "END"),
    )
    assert resources.to_sbatch_args() == [
        "--partition",
        "gpu",
        "--account",
        "my-account",
        "--time",
        "01:00:00",
        "--cpus-per-task",
        "4",
        "--mem",
        "16G",
        "--gres",
        "gpu:1",
        "--nodes",
        "1",
        "--qos",
        "normal",
        "--constraint",
        "a100",
        "--mail-type",
        "END",
    ]


def test_rewrite_host_preserves_port() -> None:
    assert (
        _rewrite_host("http://0.0.0.0:8080", advertised_host="head.example.com")
        == "http://head.example.com:8080"
    )


def test_rewrite_host_without_port() -> None:
    assert (
        _rewrite_host("http://0.0.0.0", advertised_host="head.example.com")
        == "http://head.example.com"
    )


def test_slurm_worker_backend_requires_advertised_host() -> None:
    with pytest.raises(TypeError, match="advertised_host"):
        SlurmWorkerBackend()  # ty: ignore[missing-argument]


@dataclass(frozen=True, slots=True)
class SbatchCall:
    args: list[str]
    cwd: str


@dataclass(frozen=True, slots=True)
class SlurmInvocation:
    args: list[str]


@dataclass(frozen=True, slots=True)
class FakeSlurm:
    state_dir: Path
    bin_dir: Path

    @property
    def sbatch_log(self) -> Path:
        return self.state_dir / "sbatch.json"

    @property
    def squeue_log(self) -> Path:
        return self.state_dir / "squeue.json"

    @property
    def scancel_log(self) -> Path:
        return self.state_dir / "scancel.json"

    @property
    def queue_file(self) -> Path:
        return self.state_dir / "queue.txt"

    def read_sbatch_calls(self) -> list[SbatchCall]:
        if not self.sbatch_log.exists():
            return []
        raw = json.loads(self.sbatch_log.read_text())
        return [
            SbatchCall(
                args=[str(arg) for arg in entry["args"]],
                cwd=str(entry["cwd"]),
            )
            for entry in raw
        ]

    def read_squeue_calls(self) -> list[SlurmInvocation]:
        if not self.squeue_log.exists():
            return []
        raw = json.loads(self.squeue_log.read_text())
        return [SlurmInvocation(args=[str(a) for a in entry["args"]]) for entry in raw]

    def read_scancel_calls(self) -> list[SlurmInvocation]:
        if not self.scancel_log.exists():
            return []
        raw = json.loads(self.scancel_log.read_text())
        return [SlurmInvocation(args=[str(a) for a in entry["args"]]) for entry in raw]

    def queued_jobs(self) -> list[str]:
        if not self.queue_file.exists():
            return []
        return [line for line in self.queue_file.read_text().splitlines() if line]

    def set_queued_jobs(self, job_ids: list[str]) -> None:
        self.queue_file.write_text("\n".join(job_ids))


_SBATCH_SCRIPT = """\
#!{python}
import json
import os
import pathlib
import sys

state_dir = pathlib.Path(os.environ["FAKE_SLURM_STATE"])
state_dir.mkdir(parents=True, exist_ok=True)
log = state_dir / "sbatch.json"
calls = json.loads(log.read_text()) if log.exists() else []
calls.append({{
    "args": sys.argv[1:],
    "cwd": os.getcwd(),
}})
log.write_text(json.dumps(calls))

next_id_file = state_dir / "next_id.txt"
next_id = int(next_id_file.read_text()) if next_id_file.exists() else 1000
next_id_file.write_text(str(next_id + 1))

queue_file = state_dir / "queue.txt"
queued = queue_file.read_text().splitlines() if queue_file.exists() else []
queued = [line for line in queued if line]
queued.append(str(next_id))
queue_file.write_text("\\n".join(queued))

print(next_id)
"""

_SQUEUE_SCRIPT = """\
#!{python}
import json
import os
import pathlib
import sys

state_dir = pathlib.Path(os.environ["FAKE_SLURM_STATE"])
log = state_dir / "squeue.json"
calls = json.loads(log.read_text()) if log.exists() else []
calls.append({{"args": sys.argv[1:]}})
log.write_text(json.dumps(calls))

queue_file = state_dir / "queue.txt"
queued = queue_file.read_text().splitlines() if queue_file.exists() else []
queued = [line for line in queued if line]

requested: list[str] = []
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--jobs" and i + 1 < len(args):
        requested = args[i + 1].split(",")
        break
    i += 1

for job_id in queued:
    if not requested or job_id in requested:
        print(job_id)
"""

_SCANCEL_SCRIPT = """\
#!{python}
import json
import os
import pathlib
import sys

state_dir = pathlib.Path(os.environ["FAKE_SLURM_STATE"])
log = state_dir / "scancel.json"
calls = json.loads(log.read_text()) if log.exists() else []
calls.append({{"args": sys.argv[1:]}})
log.write_text(json.dumps(calls))

queue_file = state_dir / "queue.txt"
queued = queue_file.read_text().splitlines() if queue_file.exists() else []
queued = [line for line in queued if line]
to_cancel = set(sys.argv[1:])
remaining = [j for j in queued if j not in to_cancel]
queue_file.write_text("\\n".join(remaining))
"""


def _write_script(path: Path, body: str) -> None:
    path.write_text(body.format(python=sys.executable))
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSlurm]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    _write_script(bin_dir / "sbatch", _SBATCH_SCRIPT)
    _write_script(bin_dir / "squeue", _SQUEUE_SCRIPT)
    _write_script(bin_dir / "scancel", _SCANCEL_SCRIPT)

    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_dir))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    yield FakeSlurm(state_dir=state_dir, bin_dir=bin_dir)


def test_start_pool_submits_one_sbatch_per_worker(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    backend = SlurmWorkerBackend(
        n_workers=2,
        log_dir=tmp_path / "worker-logs",
        chdir=tmp_path,
        advertised_host="head.example.com",
    )

    pool = backend.start_pool(
        server_url="http://0.0.0.0:8080",
        auth_token="secret-token",
    )

    try:
        calls = fake_slurm.read_sbatch_calls()
        assert len(calls) == 2
        assert pool.job_ids == ("1000", "1001")
    finally:
        pool.join(timeout=1.0)


def test_start_pool_passes_expected_sbatch_flags(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    log_dir = tmp_path / "worker-logs"
    chdir = tmp_path / "run-here"
    chdir.mkdir()
    backend = SlurmWorkerBackend(
        n_workers=1,
        log_dir=log_dir,
        chdir=chdir,
        advertised_host="head.example.com",
        resources=SlurmResources(partition="gpu", cpus_per_task=2),
    )

    pool = backend.start_pool(
        server_url="http://0.0.0.0:8080",
        auth_token="secret-token",
    )

    try:
        call = fake_slurm.read_sbatch_calls()[0]
        args = call.args
        assert args[0] == "--parsable"
        assert "--chdir" in args
        assert args[args.index("--chdir") + 1] == str(chdir.resolve())
        assert "--output" in args
        assert "--error" in args
        output_path = args[args.index("--output") + 1]
        error_path = args[args.index("--error") + 1]
        assert output_path.startswith(str(log_dir.resolve()))
        assert error_path.startswith(str(log_dir.resolve()))
        assert "--job-name" in args
        assert "--partition" in args
        assert "--cpus-per-task" in args
        assert "--wrap" in args
        wrap_value = args[args.index("--wrap") + 1]
        assert "-m furu.worker.cli" in wrap_value
        assert "--server-url http://head.example.com:8080" in wrap_value
        assert "--auth-token secret-token" in wrap_value
        assert not any(arg.startswith("--export=") for arg in args)
    finally:
        pool.join(timeout=1.0)


def test_start_pool_creates_log_directory_before_submission(
    fake_slurm: FakeSlurm, tmp_path: Path
) -> None:
    log_dir = tmp_path / "nested" / "log-dir"
    assert not log_dir.exists()
    backend = SlurmWorkerBackend(
        n_workers=1,
        log_dir=log_dir,
        chdir=tmp_path,
        advertised_host="head.example.com",
    )

    pool = backend.start_pool(
        server_url="http://0.0.0.0:8080",
        auth_token="secret-token",
    )

    try:
        assert log_dir.is_dir()
    finally:
        pool.join(timeout=1.0)


def test_start_pool_raises_when_sbatch_returns_no_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    _write_script(
        bin_dir / "sbatch",
        "#!{python}\nimport sys\nsys.exit(0)\n",
    )
    _write_script(bin_dir / "squeue", _SQUEUE_SCRIPT)
    _write_script(bin_dir / "scancel", _SCANCEL_SCRIPT)
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_dir))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    backend = SlurmWorkerBackend(
        n_workers=1,
        log_dir=tmp_path / "log-dir",
        chdir=tmp_path,
        advertised_host="head.example.com",
    )

    with pytest.raises(RuntimeError, match="did not return a job id"):
        backend.start_pool(
            server_url="http://0.0.0.0:8080",
            auth_token="secret-token",
        )


def test_start_pool_propagates_sbatch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    _write_script(
        bin_dir / "sbatch",
        "#!{python}\nimport sys\nprint('boom', file=sys.stderr)\nsys.exit(7)\n",
    )
    _write_script(bin_dir / "squeue", _SQUEUE_SCRIPT)
    _write_script(bin_dir / "scancel", _SCANCEL_SCRIPT)
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_dir))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    backend = SlurmWorkerBackend(
        n_workers=1,
        log_dir=tmp_path / "log-dir",
        chdir=tmp_path,
        advertised_host="head.example.com",
    )

    with pytest.raises(subprocess.CalledProcessError):
        backend.start_pool(
            server_url="http://0.0.0.0:8080",
            auth_token="secret-token",
        )


def test_is_healthy_returns_true_while_jobs_remain_queued(
    fake_slurm: FakeSlurm,
) -> None:
    fake_slurm.set_queued_jobs(["1000", "1001"])
    pool = SlurmWorkerPool(
        job_ids=("1000", "1001"),
        squeue_executable="squeue",
        scancel_executable="scancel",
        health_check_interval=0.0,
    )
    assert pool.is_healthy() is True


def test_is_healthy_returns_false_when_all_jobs_have_left_squeue(
    fake_slurm: FakeSlurm,
) -> None:
    fake_slurm.set_queued_jobs([])
    pool = SlurmWorkerPool(
        job_ids=("1000",),
        squeue_executable="squeue",
        scancel_executable="scancel",
        health_check_interval=0.0,
    )
    assert pool.is_healthy() is False


def test_is_healthy_caches_within_interval(fake_slurm: FakeSlurm) -> None:
    fake_slurm.set_queued_jobs(["1000"])
    pool = SlurmWorkerPool(
        job_ids=("1000",),
        squeue_executable="squeue",
        scancel_executable="scancel",
        health_check_interval=60.0,
    )
    assert pool.is_healthy() is True
    fake_slurm.set_queued_jobs([])
    assert pool.is_healthy() is True
    assert len(fake_slurm.read_squeue_calls()) == 1


def test_join_cancels_remaining_jobs_after_timeout(fake_slurm: FakeSlurm) -> None:
    fake_slurm.set_queued_jobs(["1000", "1001"])
    pool = SlurmWorkerPool(
        job_ids=("1000", "1001"),
        squeue_executable="squeue",
        scancel_executable="scancel",
        health_check_interval=0.0,
    )

    pool.join(timeout=0.05)

    cancel_calls = fake_slurm.read_scancel_calls()
    assert len(cancel_calls) == 1
    assert sorted(cancel_calls[0].args) == ["1000", "1001"]
    assert fake_slurm.queued_jobs() == []


def test_join_does_not_cancel_when_jobs_already_finished(
    fake_slurm: FakeSlurm,
) -> None:
    fake_slurm.set_queued_jobs([])
    pool = SlurmWorkerPool(
        job_ids=("1000",),
        squeue_executable="squeue",
        scancel_executable="scancel",
        health_check_interval=0.0,
    )

    pool.join(timeout=1.0)

    assert fake_slurm.read_scancel_calls() == []
