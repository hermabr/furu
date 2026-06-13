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
        "REQUEUED",
        "RUNNING",
        "UNKNOWN",
    }
)

_PRUNABLE_STATES = ("CANCELLED", "COMPLETED")


def _is_failed_state(state: str) -> bool:
    return state not in _UNFINISHED_STATES and state not in _PRUNABLE_STATES


@dataclass(frozen=True, slots=True)
class SlurmWorkerPool:
    _sbatch_base_args: tuple[str, ...]
    _script_path: Path
    _max_workers: int
    _max_failed_restarts: int
    _resource_request: ResourceRequest
    _server_url: str
    _auth_token: str
    _poll_interval: float
    _client: PoolApiClient
    _stop_event: threading.Event
    _scale_thread: threading.Thread
    _job_ids: list[str]
    _failed_job_ids: list[str]

    def stop(self, *, timeout: float) -> None:
        self._stop_event.set()
        self._scale_thread.join(timeout=timeout)

        deadline = time.monotonic() + timeout
        while self._active_job_ids() and time.monotonic() < deadline:
            time.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))
        if self._active_job_ids():
            subprocess.run(
                ["scancel", *self._job_ids],
                check=False,
                capture_output=True,
                text=True,
            )

    def _scale_once(self) -> dict[str, str]:
        active_job_ids = self._active_job_ids()
        states = self._task_states()
        self._failed_job_ids[:] = sorted(
            set(self._failed_job_ids)
            | {job_id for job_id, state in states.items() if _is_failed_state(state)}
        )
        self._job_ids[:] = [
            job_id
            for job_id in self._job_ids
            if job_id in active_job_ids
            or states.get(job_id) not in (None, *_PRUNABLE_STATES)
        ]
        remaining_starts = (
            self._max_workers
            + self._max_failed_restarts
            - len(self._failed_job_ids)
            - len(self._job_ids)
        )
        if len(self._job_ids) >= self._max_workers or remaining_starts <= 0:
            return states

        to_spawn = min(
            max(
                0,
                self._client.count_satisfiable_jobs(
                    resources=self._resource_request,
                    max_workers=self._max_workers,
                )
                - len(self._job_ids),
            ),
            remaining_starts,
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
        return states

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
                "-X",
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
            allocation_job_id, separator, _step_id = job_id.partition(".")
            if separator and allocation_job_id in known_job_ids:
                continue
            if job_id not in known_job_ids:
                raise ValueError(f"unexpected Slurm job id: {job_id!r}")
            states[job_id] = state.upper().split(maxsplit=1)[0].removesuffix("+")
        return states

    def _scale_loop(self) -> None:
        try:
            if self._stop_event.is_set():
                return
            self._scale_once()
            while not self._stop_event.wait(timeout=self._poll_interval):
                states = self._scale_once()
                if any(_is_failed_state(state) for state in states.values()):
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
            logger.exception(
                "failed to report slurm worker pool failure to execution coordinator"
            )
