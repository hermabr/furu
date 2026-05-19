from __future__ import annotations

import subprocess
import time


class SlurmWorkerPool:
    def __init__(
        self,
        *,
        array_job_id: str | None,
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
        if self.array_job_id is None:
            return True
        return self._unfinished_task_ids() == set(range(self.n_workers))

    def join(self, *, timeout: float) -> None:
        if self.array_job_id is None:
            return
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
        if self.array_job_id is None:
            return set()

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
