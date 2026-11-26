from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from furu import slurm
from furu.logging import get_logger
from furu.utils import format_duration
from furu.worker._cli import MANAGER_URL_ENV_VAR
from furu.worker.messages import ErrorMessage, Pool, SlurmPool, WarningMessage

if TYPE_CHECKING:
    from furu.execution.manager import Manager

logger = get_logger("slurm")

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
        self.definition: Pool = definition
        self._manager = manager
        self._key = key
        self._use_job_arrays = definition.use_job_arrays
        self._job_name = definition.job_name
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
            f"    --idle-timeout {definition.idle_timeout_seconds}\n"
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
        try:
            pending = [
                job_id
                for job_id, state in self._active_job_states().items()
                if state == "PENDING"
            ]
            if pending:
                slurm.run("scancel", *pending)
        except subprocess.SubprocessError as exc:
            logger.warning("could not cancel queued workers: %s", exc)

    def _scale_once(self) -> None:
        active_job_states = self._active_job_states()
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

        # Scripts that share the pool can name their workers differently.
        # Without a spec that waits, as the pool's first script named them.
        job_name = self._manager.job_name(self._key) or self._job_name
        for _ in range(1 if self._use_job_arrays else to_spawn):
            if self._stop_event.is_set():
                return
            # One that times out may be queued anyway; such a worker still
            # serves this pool, but goes uncounted until its idle timeout.
            submitted = slurm.run(
                "sbatch",
                "--parsable",
                *((f"--array=0-{to_spawn - 1}",) if self._use_job_arrays else ()),
                f"--job-name={job_name}",
                *self._sbatch_args,
                str(self._script_path),
            )
            job_id = submitted.strip().split(";", maxsplit=1)[0]
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
        slurm.run("scancel", *to_cancel)
        cancelled = set(to_cancel)
        self._job_ids[:] = [
            job_id for job_id in self._job_ids if job_id not in cancelled
        ]

    def _allocations(self) -> str:
        return ",".join(sorted({job_id.partition("_")[0] for job_id in self._job_ids}))

    def _active_job_states(self) -> dict[str, str]:
        """What squeue says of each worker still queued or running."""
        if not self._job_ids:
            return {}
        output = slurm.run(
            "squeue",
            "--noheader",
            "--jobs",
            self._allocations(),
            *(
                ("--array", "--format=%i %T")
                if self._use_job_arrays
                else ("--format=%A %T",)
            ),
        )
        return dict(line.split() for line in output.splitlines())

    def _task_states(self) -> dict[str, str]:
        """What accounting says of each worker, such as why it left squeue.

        Lines for anything else, such as array tasks this pool did not submit
        or the grouped row of still-queued tasks, are skipped.
        """
        if not self._job_ids:
            return {}
        output = slurm.run(
            "sacct",
            "-X",
            "--noheader",
            "-o",
            "JobID,State",
            "--parsable2",
            "-j",
            self._allocations(),
        )
        known = set(self._job_ids)
        rows = (line.split("|") for line in output.splitlines())
        # "CANCELLED by <uid>" is CANCELLED.
        return {
            job_id: state.partition(" ")[0] for job_id, state in rows if job_id in known
        }

    def _scale_loop(self) -> None:
        # Slurm not answering only delays the pool. Any other failed tick fails
        # the scripts using the pool now, but the pool keeps its workers and
        # retries, so it still serves the scripts that come later.
        while not self._stop_event.is_set():
            try:
                self._scale_once()
            except subprocess.CalledProcessError as exc:
                logger.warning("%s; retrying on the next tick: %s", exc, exc.stderr)
            except subprocess.TimeoutExpired as exc:
                logger.warning("%s; retrying on the next tick", exc)
            except Exception as exc:
                logger.exception("scale tick failed; retrying on the next tick")
                self._manager.notify(
                    self._key,
                    ErrorMessage(
                        reason="slurm worker pool scale tick failed: "
                        + traceback.format_exception_only(type(exc), exc)[-1].strip()
                    ),
                )
            self._stop_event.wait(timeout=_POLL_INTERVAL_S)
