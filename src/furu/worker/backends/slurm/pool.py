from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from furu.execution.api import WorkerPoolApiClient
from furu.resources import ResourceRequest
from furu.worker.backends import count_workers_to_launch
from furu.worker.backends.scaling import PeriodicScaler


class SlurmWorkerPool:
    def __init__(
        self,
        *,
        sbatch_base_args: tuple[str, ...],
        script_path: Path,
        max_workers: int,
        resource_request: ResourceRequest,
        client: WorkerPoolApiClient,
        poll_interval: float,
    ) -> None:
        self._sbatch_base_args = sbatch_base_args
        self._script_path = script_path
        self._max_workers = max_workers
        self._resource_request = resource_request
        self._client = client
        self._poll_interval = poll_interval
        self._array_jobs: list[tuple[str, int]] = []
        self._lock = threading.Lock()
        self._scaler = PeriodicScaler(
            interval=poll_interval,
            scale_once=self.scale,
            report_failure=lambda message: self._client.fail(message=message),
            thread_name="furu-slurm-worker-pool-scaler",
        )

    @property
    def health_check_interval(self) -> float:
        return self._poll_interval

    @property
    def n_workers(self) -> int:
        with self._lock:
            return sum(n for _, n in self._array_jobs)

    @property
    def array_job_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(array_job_id for array_job_id, _ in self._array_jobs)

    def start(self) -> None:
        self._scaler.start()

    def scale(self) -> None:
        with self._lock:
            to_spawn = count_workers_to_launch(
                self._client,
                current_workers=sum(n for _, n in self._array_jobs),
                max_workers=self._max_workers,
                resource_request=self._resource_request,
            )
            if to_spawn == 0:
                return
            result = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    *self._sbatch_base_args,
                    f"--array=0-{to_spawn - 1}",
                    str(self._script_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            array_job_id = result.stdout.strip().split(";", maxsplit=1)[0]
            self._array_jobs.append((array_job_id, to_spawn))

    def is_healthy(self) -> bool:
        array_jobs = self._array_jobs_snapshot()
        unfinished_task_ids = self._unfinished_task_ids(array_jobs)
        return (
            all(
                unfinished_task_ids[array_job_id] == set(range(n_tasks))
                for array_job_id, n_tasks in array_jobs
            )
            and self._scaler.is_healthy()
        )

    def join(self, *, timeout: float) -> None:
        self._scaler.stop(timeout=timeout)
        deadline = time.monotonic() + timeout
        while self._has_unfinished() and time.monotonic() < deadline:
            poll_interval = self._poll_interval if self._poll_interval > 0 else 0.1
            time.sleep(min(poll_interval, deadline - time.monotonic()))
        if self._has_unfinished():
            subprocess.run(
                ["scancel", *self.array_job_ids],
                check=False,
                capture_output=True,
                text=True,
            )

    def _has_unfinished(self) -> bool:
        return any(self._unfinished_task_ids(self._array_jobs_snapshot()).values())

    def _array_jobs_snapshot(self) -> tuple[tuple[str, int], ...]:
        with self._lock:
            return tuple(self._array_jobs)

    def _unfinished_task_ids(
        self, array_jobs: tuple[tuple[str, int], ...]
    ) -> dict[str, set[int]]:
        unfinished_task_ids: dict[str, set[int]] = {
            array_job_id: set() for array_job_id, _ in array_jobs
        }
        if not array_jobs:
            return unfinished_task_ids

        result = subprocess.run(
            [
                "sacct",
                "-o",
                "JobID,State,NodeList",
                "--parsable2",
                "-j",
                ",".join(array_job_id for array_job_id, _ in array_jobs),
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
