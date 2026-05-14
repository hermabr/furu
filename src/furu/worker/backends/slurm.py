from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


_LOCAL_MANAGER_HOSTS = {"0.0.0.0", "127.0.0.1", "localhost", "::", "::1"}


@dataclass(frozen=True, slots=True)
class SlurmResources:
    cpus_per_task: int | None = None
    memory: str | None = None
    time: str | None = None
    partition: str | None = None


@dataclass(frozen=True, slots=True)
class SlurmWorkerBackend:
    n_workers: int = 1
    log_dir: Path = Path("furu-slurm-logs")
    resources: SlurmResources = field(default_factory=SlurmResources)
    job_name: str = "furu-worker"
    chdir: Path = field(default_factory=Path.cwd)
    allow_local_manager_url: bool = False

    def start_pool(self, *, server_url: str, auth_token: str) -> SlurmWorkerPool:
        if self.n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        if not self.allow_local_manager_url:
            _raise_for_local_manager_url(server_url)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        job_ids = [
            _submit_worker(
                server_url=server_url,
                auth_token=auth_token,
                log_dir=self.log_dir,
                resources=self.resources,
                job_name=self.job_name,
                chdir=self.chdir,
            )
            for _ in range(self.n_workers)
        ]
        return SlurmWorkerPool(job_ids=job_ids)


class SlurmWorkerPool:
    def __init__(self, *, job_ids: list[str]) -> None:
        self._job_ids = tuple(job_ids)

    @property
    def job_ids(self) -> tuple[str, ...]:
        return self._job_ids

    def is_healthy(self) -> bool:
        if not self._job_ids:
            return False
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", ",".join(self._job_ids)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            return False

        live_job_ids = {
            line.split(maxsplit=1)[0]
            for line in result.stdout.splitlines()
            if line.strip()
        }
        return set(self._job_ids).issubset(live_job_ids)

    def join(self, *, timeout: float) -> None:
        del timeout
        if not self._job_ids:
            return
        subprocess.run(["scancel", *self._job_ids], check=False)


def _submit_worker(
    *,
    server_url: str,
    auth_token: str,
    log_dir: Path,
    resources: SlurmResources,
    job_name: str,
    chdir: Path,
) -> str:
    worker_command = shlex.join(
        [
            sys.executable,
            "-m",
            "furu.worker.cli",
            "--server-url",
            server_url,
            "--auth-token",
            auth_token,
        ]
    )
    command = [
        "sbatch",
        "--parsable",
        "--chdir",
        os.fspath(chdir),
        "--output",
        os.fspath(log_dir / "furu-worker-%j.out"),
        "--error",
        os.fspath(log_dir / "furu-worker-%j.err"),
        "--job-name",
        job_name,
    ]
    if resources.cpus_per_task is not None:
        command.extend(["--cpus-per-task", str(resources.cpus_per_task)])
    if resources.memory is not None:
        command.extend(["--mem", resources.memory])
    if resources.time is not None:
        command.extend(["--time", resources.time])
    if resources.partition is not None:
        command.extend(["--partition", resources.partition])
    command.extend(["--wrap", worker_command])

    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip().split(";", maxsplit=1)[0]


def _raise_for_local_manager_url(server_url: str) -> None:
    host = urlparse(server_url).hostname
    if host in _LOCAL_MANAGER_HOSTS:
        raise ValueError(
            "Slurm workers require a routable manager URL; pass advertised_host "
            "to Manager.run or set allow_local_manager_url=True"
        )
