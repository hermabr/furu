from __future__ import annotations

import subprocess
import time
from pathlib import Path

from furu.execution.api import ManagerApiClient
from furu.resources import ResourceRequest
from furu.worker.backends import count_workers_to_launch


class SlurmWorkerPool:
    def __init__(
        self,
        *,
        sbatch_base_args: tuple[str, ...],
        script_path: Path,
        max_workers: int,
        resource_request: ResourceRequest,
        client: ManagerApiClient,
        poll_interval: float,
    ) -> None:
        self._sbatch_base_args = sbatch_base_args
        self._script_path = script_path
        self._max_workers = max_workers
        self._resource_request = resource_request
        self._client = client
        self._poll_interval = poll_interval
        self._array_jobs: list[tuple[str, int]] = []

    @property
    def health_check_interval(self) -> float:
        return self._poll_interval

    @property
    def n_workers(self) -> int:
        return sum(n for _, n in self._array_jobs)

    @property
    def array_job_ids(self) -> tuple[str, ...]:
        return tuple(array_job_id for array_job_id, _ in self._array_jobs)

    def scale(self) -> None:
        to_spawn = count_workers_to_launch(
            self._client,
            current_workers=self.n_workers,
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
        return all(
            self._unfinished_task_ids(array_job_id) == set(range(n_tasks))
            for array_job_id, n_tasks in self._array_jobs
        )

    def join(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self._has_unfinished() and time.monotonic() < deadline:
            time.sleep(min(self._poll_interval, deadline - time.monotonic()))
        if self._has_unfinished():
            for array_job_id, _ in self._array_jobs:
                subprocess.run(
                    ["scancel", array_job_id],
                    check=False,
                    capture_output=True,
                    text=True,
                )

    def _has_unfinished(self) -> bool:
        return any(
            self._unfinished_task_ids(array_job_id)
            for array_job_id, _ in self._array_jobs
        )

    def _unfinished_task_ids(self, array_job_id: str) -> set[int]:
        result = subprocess.run(
            [
                "sacct",
                "-o",
                "JobID,State,NodeList",
                "--parsable2",
                "-j",
                array_job_id,
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
            line_array_job_id, separator, task_id = job_id.partition("_")
            if line_array_job_id != array_job_id or not separator:
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
