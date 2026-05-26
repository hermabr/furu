from __future__ import annotations

import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from furu.execution.api import PoolApiClient
from furu.logging import get_logger
from furu.resources import ResourceRequest

logger = get_logger()

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


@dataclass(slots=True)
class SlurmWorkerPool:
    _sbatch_base_args: tuple[str, ...]
    _script_path: Path
    _max_workers: int
    _resource_request: ResourceRequest
    _server_url: str
    _auth_token: str
    _poll_interval: float
    _client: PoolApiClient
    _stop_event: threading.Event
    _scale_thread: threading.Thread
    _job_ids: list[str]

    @classmethod
    def start(
        cls,
        *,
        sbatch_base_args: tuple[str, ...],
        script_path: Path,
        max_workers: int,
        resource_request: ResourceRequest,
        server_url: str,
        auth_token: str,
        poll_interval: float,
    ) -> SlurmWorkerPool:
        pool_holder: list[SlurmWorkerPool] = []
        pool = cls(
            _sbatch_base_args=sbatch_base_args,
            _script_path=script_path,
            _max_workers=max_workers,
            _resource_request=resource_request,
            _server_url=server_url,
            _auth_token=auth_token,
            _poll_interval=poll_interval,
            _client=PoolApiClient(server_url=server_url, auth_token=auth_token),
            _stop_event=threading.Event(),
            _scale_thread=threading.Thread(
                target=lambda: pool_holder[0]._scale_loop(),
                name="furu-slurm-worker-pool-scale",
            ),
            _job_ids=[],
        )
        pool_holder.append(pool)
        pool._scale_thread.start()
        return pool

    def stop(self, *, timeout: float) -> None:
        self._stop_event.set()
        self._scale_thread.join(timeout=timeout)

        deadline = time.monotonic() + timeout
        while self._has_unfinished() and time.monotonic() < deadline:
            time.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))
        if self._has_unfinished():
            subprocess.run(
                ["scancel", *self._job_ids],
                check=False,
                capture_output=True,
                text=True,
            )

    def _scale_once(self) -> None:
        if len(self._job_ids) >= self._max_workers:
            return

        to_spawn = self._client.count_satisfiable_jobs(
            resources=self._resource_request,
            max_workers=self._max_workers - len(self._job_ids),
        )
        for _ in range(to_spawn):
            result = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    *self._sbatch_base_args,
                    str(self._script_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = result.stdout.strip().split(";", maxsplit=1)[0]
            self._job_ids.append(job_id)

    def _workers_healthy(self) -> bool:
        return not any(
            state not in _UNFINISHED_STATES and state not in _HEALTHY_FINISHED_STATES
            for state in self._task_states().values()
        )

    def _has_unfinished(self) -> bool:
        return any(
            state in _UNFINISHED_STATES for state in self._task_states().values()
        )

    def _task_states(self) -> dict[str, str]:
        if not self._job_ids:
            return {}

        known_job_ids = set(self._job_ids)
        result = subprocess.run(
            [
                "sacct",
                "-o",
                "JobID,State,NodeList",
                "--parsable2",
                "-j",
                ",".join(self._job_ids),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        states: dict[str, str] = {}
        for line in result.stdout.splitlines()[1:]:
            job_id, state, _node_list = line.split("|")
            if "." in job_id or "_" in job_id:
                raise RuntimeError(
                    f"Unexpected Slurm job step in sacct output: {job_id}"
                )
            if job_id not in known_job_ids:
                raise ValueError(f"unexpected Slurm job id: {job_id!r}")
            states[job_id] = state.upper()
        return states

    def _scale_loop(self) -> None:
        try:
            if self._stop_event.is_set():
                return
            self._scale_once()
            while not self._stop_event.wait(timeout=self._poll_interval):
                self._scale_once()
                if not self._workers_healthy():
                    self._report_failure("slurm worker pool became unhealthy")
                    return
        except Exception as exc:
            self._report_failure(
                "slurm worker pool scale loop crashed: "
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )

    def _report_failure(self, message: str) -> None:
        logger.error("slurm worker pool failure: %s", message)
        try:
            self._client.fail(message=message)
        except Exception:
            logger.exception("failed to report slurm worker pool failure to manager")
