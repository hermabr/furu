from __future__ import annotations

import subprocess
import time
from pathlib import Path

from furu.execution.api import PoolApiClient
from furu.resources import ResourceRequest
from furu.worker.backends import _SelfScalingWorkerPool, count_workers_to_launch

_UNFINISHED_STATES = frozenset(
    {
        "COMPLETING",
        "PENDING",
        "PREEMPTED",
        "READY",
        "REQUEUED",
        "RUNNING",
        "UNKNOWN",
    }
)
_HEALTHY_FINISHED_STATES = frozenset({"COMPLETED"})


class SlurmWorkerPool(_SelfScalingWorkerPool):
    def __init__(
        self,
        *,
        sbatch_base_args: tuple[str, ...],
        script_path: Path,
        max_workers: int,
        resource_request: ResourceRequest,
        client: PoolApiClient,
        poll_interval: float,
    ) -> None:
        super().__init__(
            client=client,
            scale_interval=poll_interval,
            description="slurm worker pool",
        )
        self._sbatch_base_args = sbatch_base_args
        self._script_path = script_path
        self._max_workers = max_workers
        self._resource_request = resource_request
        self._poll_interval = poll_interval
        self._array_jobs: list[tuple[str, int]] = []

    @property
    def n_workers(self) -> int:
        return sum(n for _, n in self._array_jobs)

    @property
    def array_job_ids(self) -> tuple[str, ...]:
        return tuple(array_job_id for array_job_id, _ in self._array_jobs)

    def join(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        self._join_scale_loop(timeout=timeout)
        while self._has_unfinished() and time.monotonic() < deadline:
            time.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))
        if self._has_unfinished():
            subprocess.run(
                ["scancel", *self.array_job_ids],
                check=False,
                capture_output=True,
                text=True,
            )

    def _scale_once(self) -> None:
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

    def _workers_healthy(self) -> bool:
        return not any(
            state not in _UNFINISHED_STATES and state not in _HEALTHY_FINISHED_STATES
            for state in self._task_states().values()
        )

    def _has_unfinished(self) -> bool:
        return any(
            state in _UNFINISHED_STATES for state in self._task_states().values()
        )

    def _task_states(self) -> dict[tuple[str, int], str]:
        if not self._array_jobs:
            return {}

        known_array_job_ids = {array_job_id for array_job_id, _ in self._array_jobs}
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
        states: dict[tuple[str, int], str] = {}
        for line in result.stdout.splitlines()[1:]:
            job_id, state, _node_list = line.split("|")
            if "." in job_id:
                raise RuntimeError(
                    f"Unexpected Slurm job step in sacct output: {job_id}"
                )
            array_job_id, separator, task_id = job_id.partition("_")
            if array_job_id not in known_array_job_ids or not separator:
                raise ValueError(f"unexpected Slurm job id: {job_id!r}")
            if not task_id.isdecimal():
                raise RuntimeError(
                    f"Unexpected Slurm job step in sacct output: {line!r}"
                )
            states[(array_job_id, int(task_id))] = state.upper()
        return states
