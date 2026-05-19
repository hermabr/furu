from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from furu.execution.api import ManagerApiClient
from furu.resources import ResourceRequest
from furu.worker.backends.slurm.resources import SlurmResources


@dataclass(frozen=True, slots=True)
class SlurmArrayJob:
    array_job_id: str
    n_workers: int


@dataclass(frozen=True, slots=True)
class SlurmWorkerLauncher:
    chdir: Path
    worker_dir: Path
    script_path: Path
    job_name: str
    resources: SlurmResources

    def launch(self, *, n_workers: int) -> SlurmArrayJob:
        log_dir = self.worker_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--chdir={self.chdir}",
                f"--output={log_dir / 'furu-worker-%A-%a.out'}",
                f"--error={log_dir / 'furu-worker-%A-%a.err'}",
                f"--job-name={self.job_name}",
                f"--array=0-{n_workers - 1}",
                *self.resources.to_sbatch_args(),
                "--export=NIL",
                str(self.script_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        array_job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        return SlurmArrayJob(array_job_id=array_job_id, n_workers=n_workers)


class SlurmWorkerPool:
    def __init__(
        self,
        *,
        manager_server_url: str,
        auth_token: str,
        max_workers: int,
        resource_request: ResourceRequest,
        launcher: SlurmWorkerLauncher,
        poll_interval: float,
    ) -> None:
        self._client: ManagerApiClient | None = ManagerApiClient(
            manager_server_url, auth_token=auth_token
        )
        self._max_workers = max_workers
        self._resource_request: ResourceRequest | None = resource_request
        self._launcher: SlurmWorkerLauncher | None = launcher
        self._poll_interval = poll_interval
        self._jobs: list[SlurmArrayJob] = []
        self._start_available_workers()

    @classmethod
    def from_existing_jobs(
        cls, *, jobs: tuple[SlurmArrayJob, ...], poll_interval: float
    ) -> SlurmWorkerPool:
        pool = cls.__new__(cls)
        pool._client = None
        pool._max_workers = sum(job.n_workers for job in jobs)
        pool._resource_request = None
        pool._launcher = None
        pool._poll_interval = poll_interval
        pool._jobs = list(jobs)
        return pool

    @property
    def array_job_id(self) -> str | None:
        if not self._jobs:
            return None
        return self._jobs[0].array_job_id

    @property
    def array_job_ids(self) -> tuple[str, ...]:
        return tuple(job.array_job_id for job in self._jobs)

    @property
    def n_workers(self) -> int:
        return sum(job.n_workers for job in self._jobs)

    @property
    def health_check_interval(self) -> float:
        return self._poll_interval

    def is_healthy(self) -> bool:
        unfinished_task_ids = self._unfinished_task_ids()
        if any(
            unfinished_task_ids[job.array_job_id] != set(range(job.n_workers))
            for job in self._jobs
        ):
            return False
        self._start_available_workers()
        return True

    def join(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while any(self._unfinished_task_ids().values()) and time.monotonic() < deadline:
            time.sleep(min(self._poll_interval, deadline - time.monotonic()))
        if any(self._unfinished_task_ids().values()):
            subprocess.run(
                ["scancel", *self.array_job_ids],
                check=False,
                capture_output=True,
                text=True,
            )

    def _start_available_workers(self) -> None:
        if (
            self._client is None
            or self._launcher is None
            or self._resource_request is None
        ):
            return

        available_slots = self._max_workers - self.n_workers
        if available_slots <= 0:
            return

        n_workers = self._client.count_satisfiable_jobs(
            resources=self._resource_request,
            max_workers=available_slots,
        )
        n_workers = max(0, min(n_workers, available_slots))
        if n_workers == 0:
            return

        self._jobs.append(self._launcher.launch(n_workers=n_workers))

    def _unfinished_task_ids(self) -> dict[str, set[int]]:
        unfinished_task_ids: dict[str, set[int]] = {
            job.array_job_id: set() for job in self._jobs
        }
        if not self._jobs:
            return unfinished_task_ids

        result = subprocess.run(
            [
                "sacct",
                "-o",
                "JobID,State,NodeList",
                "--parsable2",
                "-j",
                ",".join(self.array_job_ids),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines()[1:]:
            job_id, state, _node_list = line.split("|")
            if "." in job_id:
                raise RuntimeError(
                    f"Unexpected Slurm job step in sacct output: {job_id}"
                )
            array_job_id, separator, task_id = job_id.partition("_")
            if array_job_id not in unfinished_task_ids or not separator:
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
                unfinished_task_ids[array_job_id].add(int(task_id))
        return unfinished_task_ids
