from __future__ import annotations

import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from furu.execution.runtime import current_executor_dir


@dataclass(frozen=True, slots=True)
class SlurmResources:
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    time_limit: str | None = None
    nodes: int | None = None
    ntasks: int | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    mem_per_cpu: str | None = None
    gres: str | None = None
    constraint: str | None = None
    extra_sbatch_args: tuple[str, ...] = ()

    def to_sbatch_args(self) -> list[str]:
        args: list[str] = []
        if self.account is not None:
            args.append(f"--account={self.account}")
        if self.partition is not None:
            args.append(f"--partition={self.partition}")
        if self.qos is not None:
            args.append(f"--qos={self.qos}")
        if self.time_limit is not None:
            args.append(f"--time={self.time_limit}")
        if self.nodes is not None:
            args.append(f"--nodes={self.nodes}")
        if self.ntasks is not None:
            args.append(f"--ntasks={self.ntasks}")
        if self.cpus_per_task is not None:
            args.append(f"--cpus-per-task={self.cpus_per_task}")
        if self.mem is not None:
            args.append(f"--mem={self.mem}")
        if self.mem_per_cpu is not None:
            args.append(f"--mem-per-cpu={self.mem_per_cpu}")
        if self.gres is not None:
            args.append(f"--gres={self.gres}")
        if self.constraint is not None:
            args.append(f"--constraint={self.constraint}")
        args.extend(self.extra_sbatch_args)
        return args


@dataclass(frozen=True, slots=True)
class SlurmWorkerBackend:
    advertised_host: str
    n_workers: int = 1
    resources: SlurmResources = field(default_factory=SlurmResources)
    log_dir: Path | str | None = None
    chdir: Path | str | None = None
    python_executable: str = sys.executable
    job_name: str = "furu-worker"
    poll_interval: float = 1.0

    def start_pool(self, *, server_url: str, auth_token: str) -> SlurmWorkerPool:
        if self.n_workers < 1:
            raise ValueError("SlurmWorkerBackend requires at least one worker")

        base_dir = Path.cwd()
        chdir = _resolve_path(self.chdir, base=base_dir)
        log_dir = (
            _resolve_path(self.log_dir, base=chdir)
            if self.log_dir is not None
            else _default_log_dir()
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        worker_server_url = _with_advertised_host(
            server_url,
            advertised_host=self.advertised_host,
        )

        job_ids: list[str] = []
        for worker_index in range(self.n_workers):
            result = subprocess.run(
                self._sbatch_command(
                    server_url=worker_server_url,
                    auth_token=auth_token,
                    chdir=chdir,
                    log_dir=log_dir,
                    worker_index=worker_index,
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            job_ids.append(_parse_sbatch_job_id(result.stdout))

        return SlurmWorkerPool(
            job_ids=job_ids,
            poll_interval=self.poll_interval,
        )

    def _sbatch_command(
        self,
        *,
        server_url: str,
        auth_token: str,
        chdir: Path,
        log_dir: Path,
        worker_index: int,
    ) -> list[str]:
        worker_command = shlex.join(
            [
                self.python_executable,
                "-m",
                "furu.worker.cli",
                "--server-url",
                server_url,
                "--auth-token",
                auth_token,
            ]
        )
        return [
            "sbatch",
            "--parsable",
            f"--chdir={chdir}",
            f"--output={log_dir / f'furu-worker-{worker_index}-%j.out'}",
            f"--error={log_dir / f'furu-worker-{worker_index}-%j.err'}",
            f"--job-name={self.job_name}",
            *self.resources.to_sbatch_args(),
            f"--wrap={worker_command}",
        ]


class SlurmWorkerPool:
    def __init__(
        self,
        *,
        job_ids: Sequence[str],
        poll_interval: float = 1.0,
    ) -> None:
        self.job_ids = tuple(job_ids)
        self._poll_interval = poll_interval

    def is_healthy(self) -> bool:
        try:
            return self._active_job_ids() == set(self.job_ids)
        except (OSError, subprocess.SubprocessError):
            return False

    def join(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                active_job_ids = self._active_job_ids()
            except (OSError, subprocess.SubprocessError):
                self.cancel()
                return

            if not active_job_ids:
                return
            if time.monotonic() >= deadline:
                self.cancel(job_ids=active_job_ids)
                return

            sleep_for = min(self._poll_interval, deadline - time.monotonic())
            if sleep_for > 0:
                time.sleep(sleep_for)

    def cancel(self, *, job_ids: set[str] | None = None) -> None:
        jobs_to_cancel = set(self.job_ids) if job_ids is None else job_ids
        if not jobs_to_cancel:
            return
        subprocess.run(
            ["scancel", *sorted(jobs_to_cancel)],
            check=False,
            capture_output=True,
            text=True,
        )

    def _active_job_ids(self) -> set[str]:
        if not self.job_ids:
            return set()

        result = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--jobs",
                ",".join(self.job_ids),
                "--format",
                "%A",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            _parse_squeue_job_id(line)
            for line in result.stdout.splitlines()
            if line.strip()
        }


def _resolve_path(path: Path | str | None, *, base: Path) -> Path:
    if path is None:
        return base.resolve()
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def _default_log_dir() -> Path:
    executor_dir = current_executor_dir()
    if executor_dir is None:
        raise RuntimeError(
            "SlurmWorkerBackend requires log_dir when it is not started by Manager.run"
        )
    return executor_dir / "workers" / "logs"


def _parse_sbatch_job_id(stdout: str) -> str:
    lines = stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(f"sbatch returned an empty job id: {stdout!r}")
    first_line = lines[0]
    job_id = first_line.split(";", maxsplit=1)[0]
    if not job_id:
        raise RuntimeError(f"sbatch returned an empty job id: {stdout!r}")
    return job_id


def _parse_squeue_job_id(line: str) -> str:
    return line.strip().split(maxsplit=1)[0]


def _with_advertised_host(server_url: str, *, advertised_host: str) -> str:
    parsed = urlsplit(server_url)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"Slurm manager server URL must include a host: {server_url}")

    return urlunsplit(
        (
            parsed.scheme,
            _format_netloc(advertised_host, port=parsed.port),
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def _format_netloc(host: str, *, port: int | None) -> str:
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if port is None:
        return netloc
    return f"{netloc}:{port}"
