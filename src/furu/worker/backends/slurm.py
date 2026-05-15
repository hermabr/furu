from __future__ import annotations

import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from furu.utils import write_private_file


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
    n_workers: int
    resources: SlurmResources
    job_name: str = "furu-worker"
    poll_interval: float = 10.0
    advertised_host: str | None = None
    allow_local_manager_url: bool = False

    def start_pool(
        self,
        *,
        server_url: str,
        auth_token: str,
        executor_dir: Path,
    ) -> SlurmWorkerPool:
        server_url = _manager_url_for_workers(
            server_url,
            advertised_host=self.advertised_host,
        )
        _validate_manager_url(
            server_url,
            allow_local_manager_url=self.allow_local_manager_url,
        )

        chdir = Path.cwd().resolve()
        worker_dir = executor_dir.resolve() / "workers"
        worker_dir.mkdir(parents=True, exist_ok=True)

        token_file = worker_dir / f"worker-{secrets.token_hex(16)}.token"
        write_private_file(token_file, auth_token, mode=0o600)

        array_job_id = self._launch_jobs(
            chdir=chdir,
            token_file=token_file,
            worker_dir=worker_dir,
            server_url=server_url,
        )

        return SlurmWorkerPool(
            array_job_id=array_job_id,
            n_workers=self.n_workers,
            poll_interval=self.poll_interval,
        )

    def _launch_jobs(
        self, chdir: Path, token_file: Path, worker_dir: Path, server_url: str
    ) -> str:
        script_path = self._write_sbatch_script(
            worker_dir=worker_dir, token_file=token_file, server_url=server_url
        )

        log_dir = worker_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--chdir={chdir}",
                f"--output={log_dir / 'furu-worker-%A-%a.out'}",
                f"--error={log_dir / 'furu-worker-%A-%a.err'}",
                f"--job-name={self.job_name}",
                f"--array=0-{self.n_workers - 1}",
                *self.resources.to_sbatch_args(),
                "--export=NIL",
                str(script_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        array_job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        return array_job_id

    def _write_sbatch_script(
        self, *, worker_dir: Path, token_file: Path, server_url: str
    ) -> Path:
        scripts_dir = worker_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"worker-{secrets.token_hex(16)}.sh"
        write_private_file(
            script_path,
            (
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "\n"
                f"exec {shlex.quote(sys.executable)} -m furu.worker.cli \\\n"
                f"    --server-url {shlex.quote(server_url)} \\\n"
                f"    --auth-token-file {shlex.quote(str(token_file))}\n"
            ),
            mode=0o700,
        )
        return script_path


class SlurmWorkerPool:
    def __init__(
        self,
        *,
        array_job_id: str,
        n_workers: int,
        poll_interval: float,
    ) -> None:
        self.array_job_id = array_job_id
        self.n_workers = n_workers
        self._poll_interval = poll_interval

    @property
    def health_check_interval(self) -> float:
        return self._poll_interval

    def is_healthy(self) -> bool:
        return self._unfinished_task_ids() == set(range(self.n_workers))

    def join(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self._unfinished_task_ids() and time.monotonic() < deadline:
            time.sleep(min(self._poll_interval, deadline - time.monotonic()))
        if self._unfinished_task_ids():
            subprocess.run(
                ["scancel", self.array_job_id],
                check=False,
                capture_output=True,
                text=True,
            )

    def _unfinished_task_ids(self) -> set[int]:
        result = subprocess.run(
            [
                "sacct",
                "-o",
                "JobID,State,NodeList",
                "--parsable2",
                "-j",
                self.array_job_id,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        unfinished_task_ids: set[int] = set()
        for line in result.stdout.splitlines()[1:]:
            job_id, state, _node_list = line.split("|")
            if "." in job_id:
                raise RuntimeError(
                    f"Unexpected Slurm job step in sacct output: {job_id}"
                )
            array_job_id, separator, task_id = job_id.partition("_")
            if array_job_id != self.array_job_id or not separator:
                raise ValueError(f"unexpected Slurm job id: {job_id!r}")
            if not task_id.isdecimal():
                raise RuntimeError(
                    f"Unexpected Slurm job step in sacct output: {line!r}"
                )
            if state.upper() in {
                "COMPLETING",
                "PENDING",
                "PREEMPTED",
                "READY",
                "REQUEUED",
                "RUNNING",
                "UNKNOWN",
            }:
                unfinished_task_ids.add(int(task_id))
        return unfinished_task_ids


def _validate_manager_url(
    server_url: str,
    *,
    allow_local_manager_url: bool,
) -> None:
    hostname = urlsplit(server_url).hostname
    if hostname is None:
        raise ValueError(
            f"Slurm manager server URL must include a hostname: {server_url!r}"
        )
    if allow_local_manager_url or not _is_local_or_wildcard_host(hostname):
        return
    raise ValueError(
        "Slurm workers need a manager URL reachable from compute nodes; "
        f"{hostname!r} is local-only or a wildcard bind address. Pass "
        "SlurmWorkerBackend(advertised_host=...) with a cluster-reachable "
        "address, or set allow_local_manager_url=True only when the compute workers can "
        "deliberately reach that address."
    )


def _manager_url_for_workers(server_url: str, *, advertised_host: str | None) -> str:
    if advertised_host is None:
        return server_url
    parsed_url = urlsplit(server_url)
    if parsed_url.hostname is None:
        raise ValueError(
            f"Slurm manager server URL must include a hostname: {server_url!r}"
        )
    if parsed_url.port is None:
        raise ValueError(
            f"Slurm manager server URL must include a port: {server_url!r}"
        )
    return parsed_url._replace(
        netloc=f"{_format_url_host(advertised_host)}:{parsed_url.port}"
    ).geturl()


def _format_url_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _is_local_or_wildcard_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        address = ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified
