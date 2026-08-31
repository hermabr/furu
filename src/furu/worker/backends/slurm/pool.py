from __future__ import annotations

import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from furu.logging import _scoped_component, get_logger
from furu.resources import ResourceRequest
from furu.worker.protocol import PoolHandoff

if TYPE_CHECKING:
    from furu.execution.execution_coordinator import ExecutionCoordinator

logger = get_logger()

_SLURM_COMMAND_TIMEOUT_S = 60.0

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

_PRUNABLE_STATES = ("COMPLETED",)


def _is_failed_state(state: str) -> bool:
    return state not in _UNFINISHED_STATES and state not in _PRUNABLE_STATES


@dataclass(frozen=True, slots=True)
class SlurmWorkerPool:
    _sbatch_base_args: tuple[str, ...]
    _script_path: Path
    _max_workers: int
    _resource_request: ResourceRequest
    _poll_interval: float
    _coordinator: ExecutionCoordinator
    _stop_event: threading.Event
    _use_job_arrays: bool
    _scale_thread: threading.Thread
    _job_ids: list[str]
    _worker_files: set[Path]

    def handoff(self) -> PoolHandoff:
        with _scoped_component("slurm"):
            self._stop_event.set()
            self._scale_thread.join()
            job_ids, self._job_ids[:] = list(self._job_ids), []
            logger.info("handed off %d slurm workers", len(job_ids))
            return PoolHandoff(job_ids=job_ids, worker_files=sorted(self._worker_files))

    def stop(self, *, timeout: float) -> None:
        with _scoped_component("slurm"):
            self._stop_event.set()
            self._scale_thread.join(timeout=timeout)

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                active_job_states = self._active_job_states()
                if active_job_states is not None and not active_job_states:
                    return
                time.sleep(
                    min(self._poll_interval, max(0.0, deadline - time.monotonic()))
                )

            if not self._job_ids:
                return
            result = subprocess.run(
                ["scancel", *self._job_ids],
                check=False,
                capture_output=True,
                text=True,
                timeout=_SLURM_COMMAND_TIMEOUT_S,
            )
            if result.returncode != 0:
                logger.error(
                    "scancel failed for slurm worker jobs %s: %s",
                    ",".join(self._job_ids),
                    result.stderr.strip(),
                )

    def _scale_once(self) -> dict[str, str]:
        active_job_states = self._active_job_states()
        states = self._task_states()
        lost_job_ids = {
            job_id
            for job_id in self._job_ids
            if active_job_states is not None
            and job_id not in active_job_states
            and ((state := states.get(job_id)) is None or not _is_failed_state(state))
        }
        self._job_ids[:] = [
            job_id
            for job_id in self._job_ids
            if job_id not in lost_job_ids
            and (
                (active_job_states is not None and job_id in active_job_states)
                or states.get(job_id) not in (None, *_PRUNABLE_STATES)
            )
        ]
        demand = min(
            self._coordinator.count_satisfiable_jobs(
                resources=self._resource_request,
                max_workers=self._max_workers,
            ),
            self._max_workers,
        )
        to_spawn = demand - len(self._job_ids)
        if to_spawn <= 0:
            if to_spawn < 0:
                self._cancel_queued_workers(-to_spawn, active_job_states or {})
            return states

        for _ in range(1 if self._use_job_arrays else to_spawn):
            if self._stop_event.is_set():
                return states
            result = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    *((f"--array=0-{to_spawn - 1}",) if self._use_job_arrays else ()),
                    *self._sbatch_base_args,
                    str(self._script_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=_SLURM_COMMAND_TIMEOUT_S,
            )
            if result.returncode != 0:
                logger.warning(
                    "sbatch failed; retrying on the next scale tick: %s",
                    result.stderr.strip(),
                )
                return states
            job_id = result.stdout.strip().split(";", maxsplit=1)[0]
            if self._use_job_arrays:
                self._job_ids.extend(f"{job_id}_{arr_i}" for arr_i in range(to_spawn))
            else:
                self._job_ids.append(job_id)
        return states

    def _cancel_queued_workers(
        self, excess: int, active_job_states: dict[str, str]
    ) -> None:
        to_cancel = [
            job_id
            for job_id in reversed(self._job_ids)
            if active_job_states.get(job_id) == "PENDING"
        ][:excess]
        if not to_cancel:
            return
        try:
            result = subprocess.run(
                ["scancel", *to_cancel],
                check=False,
                capture_output=True,
                text=True,
                timeout=_SLURM_COMMAND_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logger.warning("scancel timed out; retrying on the next scale tick")
            return
        if result.returncode != 0:
            logger.warning(
                "scancel failed; retrying on the next scale tick: %s",
                result.stderr.strip(),
            )
            return
        self._job_ids[:] = [
            job_id for job_id in self._job_ids if job_id not in set(to_cancel)
        ]

    def _active_job_states(self) -> dict[str, str] | None:
        if not self._job_ids:
            return {}

        try:
            result = subprocess.run(
                [
                    "squeue",
                    "--noheader",
                    "--jobs",
                    ",".join(
                        sorted({job_id.partition("_")[0] for job_id in self._job_ids})
                    ),
                    *(
                        ("--array", "--format=%i %T")
                        if self._use_job_arrays
                        else ("--format=%A %T",)
                    ),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=_SLURM_COMMAND_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logger.warning("squeue timed out while checking slurm jobs")
            return None
        if result.returncode != 0:
            logger.debug("squeue failed while checking slurm jobs: %s", result.stderr)
            return None
        states: dict[str, str] = {}
        for line in result.stdout.splitlines():
            job_id, _, state = line.strip().partition(" ")
            if job_id:
                states[job_id] = state
        return states

    def _task_states(self) -> dict[str, str]:
        if not self._job_ids:
            return {}

        known_job_ids = set(self._job_ids)
        known_allocation_job_ids = {
            job_id.partition("_")[0] for job_id in known_job_ids
        }
        try:
            result = subprocess.run(
                [
                    "sacct",
                    "-X",
                    "--noheader",
                    "-o",
                    "JobID,State",
                    "--parsable2",
                    "-j",
                    ",".join(sorted(known_allocation_job_ids)),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=_SLURM_COMMAND_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logger.warning("sacct timed out while checking slurm jobs")
            return {}
        if result.returncode != 0:
            logger.warning("sacct failed while checking slurm jobs: %s", result.stderr)
            return {}

        states: dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                logger.warning("ignoring malformed sacct line: %r", line)
                continue
            job_id, state = parts[0], parts[1]
            allocation_job_id, separator, _step_id = job_id.partition(".")
            if separator and allocation_job_id in known_job_ids:
                continue
            if job_id not in known_job_ids:
                if self._use_job_arrays:
                    array_allocation_job_id = allocation_job_id.partition("_")[0]
                    if (
                        "[" in job_id
                        or job_id in known_allocation_job_ids
                        or array_allocation_job_id in known_allocation_job_ids
                    ):
                        continue
                logger.warning(
                    "ignoring unexpected slurm job id from sacct: %r", job_id
                )
                continue
            states[job_id] = state.upper().split(maxsplit=1)[0].removesuffix("+")
        return states

    def _scale_loop(self) -> None:
        with _scoped_component("slurm"):
            try:
                if self._stop_event.is_set():
                    return
                self._scale_once()
                while not self._stop_event.wait(timeout=self._poll_interval):
                    states = self._scale_once()
                    if failed_states := {
                        job_id: state
                        for job_id, state in states.items()
                        if _is_failed_state(state)
                    }:
                        self._report_failure(
                            "slurm worker pool became unhealthy: "
                            + ", ".join(
                                f"{job_id} {state}"
                                for job_id, state in sorted(failed_states.items())
                            )
                        )
                        return
            except Exception as exc:  # noqa: BLE001 -- fault barrier: any crash is reported
                self._report_failure(
                    "slurm worker pool scale loop crashed: "
                    + "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                )

    def _report_failure(self, message: str) -> None:
        logger.error("slurm worker pool failure: %s", message)
        self._coordinator.fail(message)
