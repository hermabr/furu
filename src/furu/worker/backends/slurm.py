from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


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
    n_workers: int
    resources: SlurmResources
    chdir: Path | str | None = None
    python_executable: str = sys.executable
    job_name: str = "furu-worker"
    poll_interval: float = 10.0

    def start_pool(
        self,
        *,
        server_url: str,
        auth_token: str,
        executor_dir: Path,
    ) -> SlurmWorkerPool:
        if self.n_workers < 1:
            raise ValueError("SlurmWorkerBackend requires at least one worker")

        base_dir = Path.cwd()
        chdir = _resolve_path(self.chdir, base=base_dir)
        log_dir = executor_dir.resolve() / "workers" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        worker_server_url = _with_advertised_host(
            server_url,
            advertised_host=self.advertised_host,
        )

        result = subprocess.run(
            self._sbatch_command(
                server_url=worker_server_url,
                auth_token=auth_token,
                chdir=chdir,
                log_dir=log_dir,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        array_job_id = _parse_sbatch_job_id(result.stdout)

        return SlurmWorkerPool(
            array_job_id=array_job_id,
            n_workers=self.n_workers,
            poll_interval=self.poll_interval,
        )

    def _sbatch_command(
        self,
        *,
        server_url: str,
        auth_token: str,
        chdir: Path,
        log_dir: Path,
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
            f"--output={log_dir / 'furu-worker-%A-%a.out'}",
            f"--error={log_dir / 'furu-worker-%A-%a.err'}",
            f"--job-name={self.job_name}",
            f"--array=0-{self.n_workers - 1}",
            *self.resources.to_sbatch_args(),
            f"--wrap={worker_command}",
        ]


class SlurmWorkerPool:
    def __init__(
        self,
        *,
        array_job_id: str,
        n_workers: int,
        poll_interval: float,
    ) -> None:
        if n_workers < 1:
            raise ValueError("SlurmWorkerPool requires at least one worker")
        self.array_job_id = array_job_id
        self.n_workers = n_workers
        self._poll_interval = poll_interval

    @property
    def job_ids(self) -> tuple[str, ...]:
        return (self.array_job_id,)

    @property
    def health_check_interval(self) -> float:
        return self._poll_interval

    def is_healthy(self) -> bool:
        try:
            return self._active_task_ids() == set(range(self.n_workers))
        except (OSError, ValueError, subprocess.SubprocessError):
            return False

    def join(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                active_task_ids = self._active_task_ids()
            except (OSError, ValueError, subprocess.SubprocessError):
                self.cancel()
                return

            if not active_task_ids:
                return
            if time.monotonic() >= deadline:
                self.cancel()
                return

            sleep_for = min(self._poll_interval, deadline - time.monotonic())
            if sleep_for > 0:
                time.sleep(sleep_for)

    def cancel(self) -> None:
        subprocess.run(
            ["scancel", self.array_job_id],
            check=False,
            capture_output=True,
            text=True,
        )

    def _active_task_ids(self) -> set[int]:
        result = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--array",
                "--jobs",
                self.array_job_id,
                "--format",
                "%A %a",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            task_id
            for line in result.stdout.splitlines()
            if line.strip()
            for job_id, task_id in [_parse_squeue_array_task(line)]
            if job_id == self.array_job_id
        }


def _resolve_path(path: Path | str | None, *, base: Path) -> Path:
    if path is None:
        return base.resolve()
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


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


def _parse_squeue_array_task(line: str) -> tuple[str, int]:
    parts = line.strip().split()
    if len(parts) != 2:
        raise ValueError(f"squeue returned an invalid array task row: {line!r}")
    job_id, task_id = parts
    return _parse_squeue_job_id(job_id), int(task_id)


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
