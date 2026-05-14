from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, urlunparse


@dataclass(frozen=True, slots=True)
class SlurmResources:
    partition: str | None = None
    account: str | None = None
    time: str | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    gres: str | None = None
    nodes: int | None = None
    qos: str | None = None
    constraint: str | None = None
    extra_sbatch_args: tuple[str, ...] = ()

    def to_sbatch_args(self) -> list[str]:
        args: list[str] = []
        if self.partition is not None:
            args += ["--partition", self.partition]
        if self.account is not None:
            args += ["--account", self.account]
        if self.time is not None:
            args += ["--time", self.time]
        if self.cpus_per_task is not None:
            args += ["--cpus-per-task", str(self.cpus_per_task)]
        if self.mem is not None:
            args += ["--mem", self.mem]
        if self.gres is not None:
            args += ["--gres", self.gres]
        if self.nodes is not None:
            args += ["--nodes", str(self.nodes)]
        if self.qos is not None:
            args += ["--qos", self.qos]
        if self.constraint is not None:
            args += ["--constraint", self.constraint]
        args += list(self.extra_sbatch_args)
        return args


def _rewrite_host(server_url: str, advertised_host: str) -> str:
    parsed = urlparse(server_url)
    netloc = (
        f"{advertised_host}:{parsed.port}"
        if parsed.port is not None
        else advertised_host
    )
    return urlunparse(parsed._replace(netloc=netloc))


@dataclass(frozen=True, slots=True, kw_only=True)
class SlurmWorkerBackend:
    advertised_host: str
    resources: SlurmResources = field(default_factory=SlurmResources)
    n_workers: int = 1
    log_dir: Path = field(default_factory=lambda: Path("logs") / "furu-workers")
    chdir: Path = field(default_factory=Path.cwd)
    python_executable: str = field(default_factory=lambda: sys.executable)
    job_name_prefix: str = "furu-worker"
    health_check_interval: float = 5.0

    def start_pool(self, *, server_url: str, auth_token: str) -> SlurmWorkerPool:
        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}")
        worker_server_url = _rewrite_host(server_url, self.advertised_host)
        log_dir = self.log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        chdir = self.chdir.resolve()

        worker_command = (
            f"{shlex.quote(self.python_executable)} -m furu.worker.cli "
            f"--server-url {shlex.quote(worker_server_url)} "
            f"--auth-token {shlex.quote(auth_token)}"
        )

        job_name = self.job_name_prefix
        stdout_path = log_dir / f"{job_name}-%A_%a.out"
        stderr_path = log_dir / f"{job_name}-%A_%a.err"
        sbatch_cmd = [
            "sbatch",
            "--parsable",
            "--array",
            f"0-{self.n_workers - 1}",
            "--chdir",
            str(chdir),
            "--job-name",
            job_name,
            "--output",
            str(stdout_path),
            "--error",
            str(stderr_path),
            *self.resources.to_sbatch_args(),
            "--wrap",
            worker_command,
        ]
        result = subprocess.run(
            sbatch_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        raw = result.stdout.strip()
        if not raw:
            raise RuntimeError(
                "sbatch --parsable did not return a job id "
                f"(stdout={result.stdout!r}, stderr={result.stderr!r})"
            )
        array_job_id = raw.split(";", 1)[0]
        if not array_job_id:
            raise RuntimeError(
                f"sbatch --parsable returned an empty job id (stdout={result.stdout!r})"
            )

        return SlurmWorkerPool(
            array_job_id=array_job_id,
            n_tasks=self.n_workers,
            health_check_interval=self.health_check_interval,
        )


class SlurmWorkerPool:
    def __init__(
        self,
        *,
        array_job_id: str,
        n_tasks: int,
        health_check_interval: float,
    ) -> None:
        self._array_job_id = array_job_id
        self._n_tasks = n_tasks
        self._expected_tasks: frozenset[str] = frozenset(
            f"{array_job_id}_{i}" for i in range(n_tasks)
        )
        self._health_check_interval = health_check_interval
        self._lock = threading.Lock()
        self._last_check_at: float | None = None
        self._last_active_tasks: frozenset[str] = self._expected_tasks

    @property
    def array_job_id(self) -> str:
        return self._array_job_id

    def _query_active_tasks(self) -> frozenset[str]:
        result = subprocess.run(
            [
                "squeue",
                "--jobs",
                self._array_job_id,
                "--array",
                "--noheader",
                "--format=%i",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        active: set[str] = set()
        for line in result.stdout.splitlines():
            token = line.strip()
            if token in self._expected_tasks:
                active.add(token)
        return frozenset(active)

    def is_healthy(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if (
                self._last_check_at is None
                or now - self._last_check_at >= self._health_check_interval
            ):
                self._last_active_tasks = self._query_active_tasks()
                self._last_check_at = now
            return bool(self._last_active_tasks)

    def join(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        active = self._query_active_tasks()
        while active and time.monotonic() < deadline:
            time.sleep(0.1)
            active = self._query_active_tasks()
        if active:
            subprocess.run(
                ["scancel", self._array_job_id],
                check=False,
                capture_output=True,
                text=True,
            )
            with self._lock:
                self._last_active_tasks = frozenset()
                self._last_check_at = time.monotonic()
