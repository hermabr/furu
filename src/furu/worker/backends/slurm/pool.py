from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from furu.config import get_config
from furu.logging import get_logger
from furu.utils import format_duration
from furu.worker._cli import MANAGER_URL_ENV_VAR
from furu.worker.protocol import ErrorMessage, SlurmPool, WarningMessage

if TYPE_CHECKING:
    from furu.execution.manager import Manager

logger = get_logger("slurm")

_SLURM_COMMAND_TIMEOUT_S = 60.0
_POLL_INTERVAL_S = 10.0
# A failing pool replaces at most one worker per minute; it is broken once
# more than 10 workers failed over at least 30 minutes with no job completing.
_REPLACE_INTERVAL_S = 60.0
_BROKEN_AFTER_FAILURES = 10
_BROKEN_AFTER_S = 30 * 60.0

# States a job never leaves. Anything else, including no state at all or one
# this code has never seen, keeps the worker owned until Slurm reports one of
# these. Preempted jobs are requeued, so PREEMPTED is not one of them.
_FINISHED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "REVOKED",
        "TIMEOUT",
    }
)


def _slurm(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=_SLURM_COMMAND_TIMEOUT_S,
    )


class SlurmWorkerPool:
    """One sbatch recipe's workers, sized to the manager's queue every 10 s."""

    def __init__(
        self,
        *,
        manager: Manager,
        key: str,
        definition: SlurmPool,
        directory: Path,
    ) -> None:
        self._manager = manager
        self._key = key
        self._use_job_arrays = definition.use_job_arrays
        self._job_ids: list[str] = []
        self._streak: list[str] = []  # "<job id> <state>" since a job last completed
        self._streak_started = 0.0
        self._last_submit = -_REPLACE_INTERVAL_S
        self._stop_event = threading.Event()

        # Workers are named after their Slurm job so `sacct -j <worker>` works.
        job_id = (
            "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            if self._use_job_arrays
            else "${SLURM_JOB_ID}"
        )
        # One file per worker, named after its job, so either side can find it.
        self._log_dir = log_dir = directory / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._script_path = directory / "worker.sh"
        self._script_path.touch(mode=0o700)
        self._script_path.chmod(0o700)
        # Rewritten in place each time a manager starts; sbatch copies the
        # script at submission, so queued workers keep their manager's URL.
        self._script_path.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            + "".join(f"{command}\n" for command in definition.pre_worker_commands)
            # Do not leak the manager's environment into the job's snapshot.
            + "unset VIRTUAL_ENV\n"
            f"export {MANAGER_URL_ENV_VAR}={shlex.quote(manager.url)}\n"
            f'component="{job_id}"\n'
            # Workers run the manager's own furu, wherever it came from.
            f"exec {shlex.quote(sys.executable)} -m furu.worker._cli \\\n"
            f"    --pool {shlex.quote(key)} \\\n"
            '    --component "${component}" \\\n'
            f'    --log {shlex.quote(str(log_dir))}/"${{component}}.log" \\\n'
            f"    --idle-timeout {get_config().worker.idle_timeout_seconds}\n"
        )
        self._sbatch_args = (
            f"--chdir={directory}",
            # stdout and stderr, merged as Slurm does by default.
            f"--output={log_dir}/{'%A_%a' if self._use_job_arrays else '%j'}.log",
            *definition.sbatch_args,
        )
        self._scale_thread = threading.Thread(
            target=self._scale_loop, name="furu-slurm-worker-pool-scale"
        )
        self._scale_thread.start()

    def job_completed(self) -> None:
        self._streak.clear()

    def stop(self) -> None:
        """Stop scaling and cancel queued workers; running ones finish."""
        self._stop_event.set()
        self._scale_thread.join()
        pending = [
            job_id
            for job_id, state in (self._active_job_states() or {}).items()
            if state == "PENDING"
        ]
        if pending:
            _slurm("scancel", *pending)

    def _scale_once(self) -> None:
        active_job_states = self._active_job_states()
        if active_job_states is None:
            return  # squeue unavailable: keep every worker and retry next tick
        # squeue decides liveness; accounting only explains why a job left it.
        states = self._task_states() | active_job_states
        finished = {
            job_id: state
            for job_id in self._job_ids
            if (state := states.get(job_id)) in _FINISHED_STATES
        }
        failed = {j: s for j, s in finished.items() if s != "COMPLETED"}
        if finished:
            logger.info(
                "forgetting finished slurm workers: %s",
                ", ".join(f"{j} {s}" for j, s in sorted(finished.items())),
            )
            self._job_ids[:] = [j for j in self._job_ids if j not in finished]
        now = time.monotonic()
        if failed:
            if not self._streak:
                self._streak_started = now
            for job_id, state in sorted(failed.items()):
                self._streak.append(f"{job_id} {state}")
                log = self._log_dir / f"{job_id}.log"
                self._manager.notify(
                    self._key,
                    WarningMessage(reason=f"slurm worker {job_id} {state} → {log}"),
                )
            if (
                len(self._streak) > _BROKEN_AFTER_FAILURES
                and now - self._streak_started >= _BROKEN_AFTER_S
            ):
                self._manager.notify(
                    self._key,
                    ErrorMessage(
                        reason=f"slurm pool is broken: {len(self._streak)} workers "
                        f"failed over {format_duration(now - self._streak_started)} "
                        f"without a job completing: {', '.join(self._streak)}"
                    ),
                )
                self._streak.clear()

        to_spawn = self._manager.demand(self._key) - len(self._job_ids)
        if to_spawn < 0:
            self._cancel_queued_workers(-to_spawn, active_job_states)
            return
        if to_spawn == 0:
            return
        if self._streak:
            if now - self._last_submit < _REPLACE_INTERVAL_S:
                return
            to_spawn = 1
            self._last_submit = now

        for _ in range(1 if self._use_job_arrays else to_spawn):
            if self._stop_event.is_set():
                return
            result = _slurm(
                "sbatch",
                "--parsable",
                *((f"--array=0-{to_spawn - 1}",) if self._use_job_arrays else ()),
                *self._sbatch_args,
                str(self._script_path),
            )
            if result.returncode != 0:
                logger.warning(
                    "sbatch failed; retrying on the next scale tick: %s",
                    result.stderr.strip(),
                )
                return
            job_id = result.stdout.strip().split(";", maxsplit=1)[0]
            if self._use_job_arrays:
                self._job_ids.extend(f"{job_id}_{arr_i}" for arr_i in range(to_spawn))
            else:
                self._job_ids.append(job_id)

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
            result = _slurm("scancel", *to_cancel)
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
            result = _slurm(
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
            result = _slurm(
                "sacct",
                "-X",
                "--noheader",
                "-o",
                "JobID,State",
                "--parsable2",
                "-j",
                ",".join(sorted(known_allocation_job_ids)),
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
        try:
            while not self._stop_event.is_set():
                self._scale_once()
                self._stop_event.wait(timeout=_POLL_INTERVAL_S)
        except Exception as exc:
            logger.exception("scale loop crashed")
            self._manager.notify(
                self._key,
                ErrorMessage(
                    reason="slurm worker pool scale loop crashed: "
                    + traceback.format_exception_only(type(exc), exc)[-1].strip()
                ),
            )
