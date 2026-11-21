from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

from furu.logging import get_logger
from furu.protocol.messages import ErrorMessage, Pool, SlurmPool, WarningMessage
from furu.utils import format_duration
from furu.worker._cli import MANAGER_URL_ENV_VAR

logger = get_logger("slurm")

_SLURM_COMMAND_TIMEOUT_S = 60.0
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


def _slurm(*argv: str) -> str | None:
    """Run a Slurm command; its output, or None once a warning says why there is none.

    Whatever failed is tried again on the pool's next tick.
    """
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=_SLURM_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        logger.warning("%s timed out", argv[0])
        return None
    if result.returncode != 0:
        logger.warning("%s failed: %s", argv[0], result.stderr.strip())
        return None
    return result.stdout


def _states(output: str, separator: str) -> dict[str, str]:
    """Each job id on a line of Slurm's output, and the state in the field after it."""
    states = {}
    for line in output.splitlines():
        job_id, _, rest = line.strip().partition(separator)
        if job_id:
            states[job_id] = rest.partition(separator)[0]
    return states


class SlurmWorkerPool:
    """One sbatch recipe's workers, whose workers connect to ``url``.

    The manager calls ``scale`` with how many it should have.
    """

    def __init__(
        self,
        *,
        url: str,
        key: str,
        definition: SlurmPool,
        directory: Path,
    ) -> None:
        self.definition: Pool = definition
        self._use_job_arrays = definition.use_job_arrays
        self._job_ids: list[str] = []
        self._streak: list[str] = []  # "<job id> <state>" since a job last completed
        self._streak_started = 0.0
        self._last_submit = -_REPLACE_INTERVAL_S

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
            f"export {MANAGER_URL_ENV_VAR}={shlex.quote(url)}\n"
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
            f"--job-name={definition.job_name}",
            *definition.sbatch_args,
        )

    def job_completed(self) -> None:
        self._streak.clear()

    def stop(self) -> None:
        """Cancel queued workers; running ones finish."""
        pending = [
            job_id
            for job_id, state in (self._active_job_states() or {}).items()
            if state == "PENDING"
        ]
        if pending:
            _slurm("scancel", *pending)

    def scale(self, demand: int) -> list[WarningMessage | ErrorMessage]:
        """Start or cancel workers so that there are ``demand``.

        Returns what the scripts using the pool should be told.
        """
        messages: list[WarningMessage | ErrorMessage] = []
        active_job_states = self._active_job_states()
        if active_job_states is None:
            return messages  # squeue unavailable: keep every worker and retry
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
                messages.append(
                    WarningMessage(reason=f"slurm worker {job_id} {state} → {log}")
                )
            if (
                len(self._streak) > _BROKEN_AFTER_FAILURES
                and now - self._streak_started >= _BROKEN_AFTER_S
            ):
                messages.append(
                    ErrorMessage(
                        reason=f"slurm pool is broken: {len(self._streak)} workers "
                        f"failed over {format_duration(now - self._streak_started)} "
                        f"without a job completing: {', '.join(self._streak)}"
                    )
                )
                self._streak.clear()

        to_spawn = demand - len(self._job_ids)
        if to_spawn < 0:
            self._cancel_queued_workers(-to_spawn, active_job_states)
            return messages
        if to_spawn == 0:
            return messages
        if self._streak:
            if now - self._last_submit < _REPLACE_INTERVAL_S:
                return messages
            to_spawn = 1
            self._last_submit = now

        for _ in range(1 if self._use_job_arrays else to_spawn):
            # One that timed out may be queued anyway; such a worker still
            # serves this pool, but goes uncounted until its idle timeout.
            submitted = _slurm(
                "sbatch",
                "--parsable",
                *((f"--array=0-{to_spawn - 1}",) if self._use_job_arrays else ()),
                *self._sbatch_args,
                str(self._script_path),
            )
            if submitted is None:
                return messages
            job_id = submitted.strip().split(";", maxsplit=1)[0]
            if self._use_job_arrays:
                self._job_ids.extend(f"{job_id}_{arr_i}" for arr_i in range(to_spawn))
            else:
                self._job_ids.append(job_id)
        return messages

    def _cancel_queued_workers(
        self, excess: int, active_job_states: dict[str, str]
    ) -> None:
        to_cancel = [
            job_id
            for job_id in reversed(self._job_ids)
            if active_job_states.get(job_id) == "PENDING"
        ][:excess]
        if not to_cancel or _slurm("scancel", *to_cancel) is None:
            return
        self._job_ids[:] = [
            job_id for job_id in self._job_ids if job_id not in set(to_cancel)
        ]

    def _allocations(self) -> str:
        return ",".join(sorted({job_id.partition("_")[0] for job_id in self._job_ids}))

    def _active_job_states(self) -> dict[str, str] | None:
        """What squeue says of each worker; None when it cannot say."""
        if not self._job_ids:
            return {}
        output = _slurm(
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
        return None if output is None else _states(output, " ")

    def _task_states(self) -> dict[str, str]:
        """What accounting says of each worker.

        Its other lines, such as job steps and array tasks this pool did not
        submit, are skipped.
        """
        if not self._job_ids:
            return {}
        output = _slurm(
            "sacct",
            "-X",
            "--noheader",
            "-o",
            "JobID,State",
            "--parsable2",
            "-j",
            self._allocations(),
        )
        return {
            job_id: state.upper().split(maxsplit=1)[0].removesuffix("+")
            for job_id, state in _states(output or "", "|").items()
            if job_id in self._job_ids and state
        }
