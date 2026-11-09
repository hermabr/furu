from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from furu.config import get_config
from furu.logging import get_logger
from furu.utils import format_duration
from furu.worker._cli import MANAGER_URL_ENV_VAR
from furu.worker.messages import ErrorMessage, SlurmPool, WarningMessage

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


@dataclass(slots=True)
class _Degraded:
    """Workers failed since a job last completed; the pool replaces them slowly.

    A completed job makes the pool healthy again. Once it is broken, the scripts
    hear so and a new streak starts: the pool stays degraded and reports again
    if the failures go on.
    """

    since: float
    failures: list[str] = field(default_factory=list)  # "<job id> <state>"
    last_submit: float = -_REPLACE_INTERVAL_S

    def fail(self, failures: list[str], now: float) -> str | None:
        """Record failed workers; say why the pool is broken, if it now is."""
        self.failures += failures
        if (
            len(self.failures) <= _BROKEN_AFTER_FAILURES
            or now - self.since < _BROKEN_AFTER_S
        ):
            return None
        reason = (
            f"slurm pool is broken: {len(self.failures)} workers failed over "
            f"{format_duration(now - self.since)} without a job completing: "
            + ", ".join(self.failures)
        )
        self.since, self.failures = now, []
        return reason

    def may_submit(self, now: float) -> bool:
        """Allow one replacement per interval."""
        if now - self.last_submit < _REPLACE_INTERVAL_S:
            return False
        self.last_submit = now
        return True


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
        self._job_ids: list[str] = []  # "<array job id>_<task id>"
        # None while healthy. Set by the scale thread, cleared by the manager's.
        self._degraded: _Degraded | None = None
        self._degraded_lock = threading.Lock()
        self._stop_event = threading.Event()

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
            # Named after its Slurm array task so `sacct -j <worker>` works.
            'component="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"\n'
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
            f"--output={log_dir}/%A_%a.log",
            *definition.sbatch_args,
        )
        self._scale_thread = threading.Thread(
            target=self._scale_loop, name="furu-slurm-worker-pool-scale"
        )
        self._scale_thread.start()

    def job_completed(self) -> None:
        with self._degraded_lock:
            self._degraded = None

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
        for job_id, state in sorted(failed.items()):
            log = self._log_dir / f"{job_id}.log"
            self._manager.notify(
                self._key,
                WarningMessage(reason=f"slurm worker {job_id} {state} → {log}"),
            )
        broken = None
        if failed:
            with self._degraded_lock:
                self._degraded = self._degraded or _Degraded(since=now)
                broken = self._degraded.fail(
                    [f"{j} {s}" for j, s in sorted(failed.items())], now
                )
        if broken is not None:
            self._manager.notify(self._key, ErrorMessage(reason=broken))

        to_spawn = self._manager.demand(self._key) - len(self._job_ids)
        if to_spawn < 0:
            self._cancel_queued_workers(-to_spawn, active_job_states)
            return
        if to_spawn == 0:
            return
        with self._degraded_lock:
            if self._degraded is not None:
                if not self._degraded.may_submit(now):
                    return
                to_spawn = 1

        if self._stop_event.is_set():
            return
        try:
            result = _slurm(
                "sbatch",
                "--parsable",
                f"--array=0-{to_spawn - 1}",
                *self._sbatch_args,
                str(self._script_path),
            )
        except subprocess.TimeoutExpired:
            # Slurm may have queued it anyway; such workers still serve this
            # pool, but go uncounted until their idle timeout.
            logger.warning("sbatch timed out; retrying on the next scale tick")
            return
        if result.returncode != 0:
            logger.warning(
                "sbatch failed; retrying on the next scale tick: %s",
                result.stderr.strip(),
            )
            return
        job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        self._job_ids.extend(f"{job_id}_{task}" for task in range(to_spawn))

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
                "--array",
                "--format=%i %T",
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

        try:
            result = _slurm(
                "sacct",
                "-X",
                "--noheader",
                "-o",
                "JobID,State",
                "--parsable2",
                "-j",
                ",".join(
                    sorted({job_id.partition("_")[0] for job_id in self._job_ids})
                ),
            )
        except subprocess.TimeoutExpired:
            logger.warning("sacct timed out while checking slurm jobs")
            return {}
        if result.returncode != 0:
            logger.warning("sacct failed while checking slurm jobs: %s", result.stderr)
            return {}

        # Only lines naming one of our tasks count: sacct also reports untracked
        # siblings and still-queued tasks grouped as "<job id>_[1-3]".
        states: dict[str, str] = {}
        known_job_ids = set(self._job_ids)
        for line in result.stdout.splitlines():
            job_id, _, state = line.partition("|")
            if job_id in known_job_ids and state.strip():
                states[job_id] = state.upper().split()[0].removesuffix("+")
        return states

    def _scale_loop(self) -> None:
        # A failed tick fails the scripts using the pool now, but the pool keeps
        # its workers and retries, so it still serves the scripts that come later.
        while not self._stop_event.is_set():
            try:
                self._scale_once()
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
