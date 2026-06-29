from __future__ import annotations

import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from furu.execution.api import PoolApiClient
from furu.logging import _scoped_component, get_logger
from furu.resources import ResourceRequest

logger = get_logger()

_SLURM_COMMAND_TIMEOUT_S = 60.0

_UNFINISHED_STATES = frozenset(
    {
        "COMPLETING",
        "PENDING",
        "PREEMPTED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
        "RUNNING",
        "UNKNOWN",
    }
)

_PRUNABLE_STATES = ("COMPLETED",)
_WORKER_LOST_STATES = frozenset(
    {
        "CANCELLED",
        "PREEMPTED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
    }
)
_SACCT_FIELD_NAMES = (
    "JobID",
    "State",
    "Restarts",
    "Start",
    "End",
    "NodeList",
    "Reason",
)


def _is_failed_state(state: str) -> bool:
    return (
        state not in _UNFINISHED_STATES
        and state not in _PRUNABLE_STATES
        and state not in _WORKER_LOST_STATES
    )


@dataclass(frozen=True, slots=True)
class SlurmTaskRecord:
    job_id: str
    state: str
    restarts: str
    start: str
    end: str
    node_list: str
    reason: str

    @property
    def details(self) -> dict[str, str]:
        return {
            "slurm_job_id": self.job_id,
            "slurm_state": self.state,
            "slurm_restarts": self.restarts,
            "slurm_start": self.start,
            "slurm_end": self.end,
            "slurm_node_list": self.node_list,
            "slurm_reason": self.reason,
        }

    @property
    def diagnostic(self) -> str:
        return (
            f"JobID={self.job_id} State={self.state} Restarts={self.restarts} "
            f"Start={self.start} End={self.end} NodeList={self.node_list} "
            f"Reason={self.reason}"
        )


def _restart_count(restarts: str) -> int:
    try:
        return int(restarts)
    except ValueError:
        return 0


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
    _use_job_arrays: bool
    _scale_thread: threading.Thread
    _job_ids: list[str]
    _failed_job_ids: list[str]
    _reported_worker_loss_restarts: dict[str, int]

    def stop(self, *, timeout: float) -> None:
        with _scoped_component("slurm"):
            self._stop_event.set()
            self._scale_thread.join(timeout=timeout)

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                active_job_ids = self._active_job_ids()
                if active_job_ids is not None and not active_job_ids:
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
        active_job_ids = self._active_job_ids()
        records = self._task_records()
        states = {job_id: record.state for job_id, record in records.items()}
        worker_loss_records = {
            job_id: record
            for job_id, record in records.items()
            if self._should_report_worker_loss(record)
        }
        lost_job_ids = {
            job_id
            for job_id in self._job_ids
            if active_job_ids is not None
            and job_id not in active_job_ids
            and ((state := states.get(job_id)) is None or not _is_failed_state(state))
        }
        self._failed_job_ids[:] = sorted(
            set(self._failed_job_ids)
            | {job_id for job_id, state in states.items() if _is_failed_state(state)}
        )
        self._job_ids[:] = [
            job_id
            for job_id in self._job_ids
            if job_id not in lost_job_ids
            and (
                (active_job_ids is not None and job_id in active_job_ids)
                or states.get(job_id) not in (None, *_PRUNABLE_STATES)
            )
        ]
        for job_id in sorted(worker_loss_records):
            record = worker_loss_records[job_id]
            self._client.worker_lost(
                worker=self._worker_name(job_id),
                reason=f"slurm worker died: {record.diagnostic}",
                details=record.details,
            )
        for job_id in sorted(lost_job_ids - worker_loss_records.keys()):
            record = records.get(job_id)
            self._client.worker_lost(
                worker=self._worker_name(job_id),
                details=record.details if record is not None else {},
                reason=(
                    f"slurm worker disappeared: {record.diagnostic}"
                    if record is not None
                    else "worker is no longer active"
                ),
            )
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
        if to_spawn <= 0:
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

    def _active_job_ids(self) -> set[str] | None:
        if not self._job_ids:
            return set()

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
                        ("--array", "--format=%i")
                        if self._use_job_arrays
                        else ("--format=%A",)
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
        return {
            line.strip().split(maxsplit=1)[0]
            for line in result.stdout.splitlines()
            if line.strip()
        }

    def _worker_name(self, job_id: str) -> str:
        allocation_job_id, separator, array_task_id = job_id.partition("_")
        return (
            f"slurm-worker-{allocation_job_id}a{array_task_id}"
            if separator
            else f"slurm-worker-{allocation_job_id}"
        )

    def _should_report_worker_loss(self, record: SlurmTaskRecord) -> bool:
        restarts = _restart_count(record.restarts)
        reported_restarts = self._reported_worker_loss_restarts.get(record.job_id, 0)
        if restarts > reported_restarts:
            self._reported_worker_loss_restarts[record.job_id] = restarts
            return True
        if record.state in _WORKER_LOST_STATES and reported_restarts == 0:
            self._reported_worker_loss_restarts[record.job_id] = max(restarts, 1)
            return True
        return False

    def _task_states(self) -> dict[str, str]:
        return {job_id: record.state for job_id, record in self._task_records().items()}

    def _task_records(self) -> dict[str, SlurmTaskRecord]:
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
                    ",".join(_SACCT_FIELD_NAMES),
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

        records: dict[str, SlurmTaskRecord] = {}
        for line in result.stdout.splitlines():
            parts = line.split("|", maxsplit=len(_SACCT_FIELD_NAMES) - 1)
            if len(parts) < 2:
                logger.warning("ignoring malformed sacct line: %r", line)
                continue
            parts = [*parts, *([""] * (len(_SACCT_FIELD_NAMES) - len(parts)))]
            job_id, state, restarts, start, end, node_list, reason = parts
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
            records[job_id] = SlurmTaskRecord(
                job_id=job_id,
                state=state.upper().split(maxsplit=1)[0].removesuffix("+"),
                restarts=restarts or "0",
                start=start or "N/A",
                end=end or "N/A",
                node_list=node_list or "N/A",
                reason=reason or "N/A",
            )
        return records

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
            except Exception as exc:
                self._report_failure(
                    "slurm worker pool scale loop crashed: "
                    + "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                )

    def _report_failure(self, message: str) -> None:
        logger.error("slurm worker pool failure: %s", message)
        try:
            self._client.fail(message=message)
        except Exception:
            logger.exception(
                "failed to report slurm worker pool failure to execution coordinator"
            )
