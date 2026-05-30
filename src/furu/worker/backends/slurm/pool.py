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
_SUCCESS_STATES = frozenset({"COMPLETED"})


@dataclass(frozen=True, slots=True)
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

    def stop(self, *, timeout: float) -> None:
        self._stop_event.set()
        self._scale_thread.join(timeout=timeout)

        deadline = time.monotonic() + timeout
        while self._has_unfinished_jobs():
            if time.monotonic() >= deadline:
                break
            time.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))
        if self._has_unfinished_jobs():
            subprocess.run(
                ["scancel", *self._job_ids],
                check=False,
                capture_output=True,
                text=True,
            )

    def _has_unfinished_jobs(self) -> bool:
        return bool(self._active_job_ids()) or any(
            state in _UNFINISHED_STATES for state in self._task_states().values()
        )

    def _scale_once(self) -> bool:
        if not self._prune_finished_jobs():
            return False
        if len(self._job_ids) >= self._max_workers:
            return True

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
        return True

    def _prune_finished_jobs(self) -> bool:
        if not self._job_ids:
            return True

        active_job_ids = self._active_job_ids()
        states = self._task_states()
        failed_states = {
            job_id: state
            for job_id, state in states.items()
            if state not in _UNFINISHED_STATES and state not in _SUCCESS_STATES
        }
        if failed_states:
            self._report_failure(
                "slurm worker pool became unhealthy: "
                + ", ".join(
                    f"{job_id}={state}"
                    for job_id, state in sorted(failed_states.items())
                )
            )
            return False

        self._job_ids[:] = [
            job_id
            for job_id in self._job_ids
            if job_id in active_job_ids or states.get(job_id) in _UNFINISHED_STATES
        ]
        return True

    def _active_job_ids(self) -> set[str]:
        if not self._job_ids:
            return set()

        result = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--jobs",
                ",".join(self._job_ids),
                "--format=%A",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            line.strip().split(maxsplit=1)[0]
            for line in result.stdout.splitlines()
            if line.strip()
        }

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
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug("sacct failed while checking slurm jobs: %s", result.stderr)
            return {}

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
            if not self._scale_once():
                return
            while not self._stop_event.wait(timeout=self._poll_interval):
                if not self._scale_once():
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
