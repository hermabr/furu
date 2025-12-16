import contextlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Self

from ..config import HULDRA_CONFIG
from ..storage import StateManager
from ..storage.state import _HuldraState


class SubmititAdapter:
    """Adapter for working with submitit executors."""

    JOB_PICKLE = "job.pkl"

    def __init__(self: Self, executor: Any):
        self.executor = executor

    def submit(self: Self, fn: Callable[[], None]) -> Any:
        """Submit a job to the executor."""
        return self.executor.submit(fn)

    def wait(self: Self, job: Any, timeout: Optional[float] = None) -> None:
        """Wait for job completion."""
        with contextlib.suppress(Exception):
            if timeout:
                job.result(timeout=timeout)
            else:
                job.wait()

    def get_job_id(self: Self, job: Any) -> Optional[str]:
        """Get job ID if available."""
        job_id = getattr(job, "job_id", None)
        if job_id:
            return str(job_id)
        return None

    def is_done(self: Self, job: Any) -> bool:
        """Check if job is done."""
        done_fn = getattr(job, "done", None)
        if done_fn and callable(done_fn):
            return done_fn()
        return False

    def get_state(self: Self, job: Any) -> Optional[str]:
        """Get job state from scheduler."""
        try:
            state_fn = getattr(job, "state", None)
            if state_fn and callable(state_fn):
                return state_fn()
        except Exception:
            pass
        return None

    def pickle_job(self: Self, job: Any, directory: Path) -> None:
        """Pickle job handle to file."""
        try:
            import cloudpickle as pickle  # type: ignore
        except Exception:  # pragma: no cover
            import pickle  # type: ignore

        job_path = directory / self.JOB_PICKLE
        with job_path.open("wb") as f:
            pickle.dump(job, f)

    def load_job(self: Self, directory: Path) -> Any:
        """Load job handle from pickle file."""
        job_path = directory / self.JOB_PICKLE
        if not job_path.is_file():
            return None
        try:
            try:
                import cloudpickle as pickle  # type: ignore
            except Exception:  # pragma: no cover
                import pickle  # type: ignore

            with job_path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def watch_job_id(
        self,
        job: Any,
        directory: Path,
        *,
        attempt_id: str,
        callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Watch for job ID in background thread and update state."""

        def watcher():
            while True:
                job_id = self.get_job_id(job)
                if job_id:

                    def mutate(state: _HuldraState) -> None:
                        attempt = state.attempt
                        if attempt is None or attempt.id != attempt_id:
                            return
                        attempt.scheduler["job_id"] = job_id

                    StateManager.update_state(directory, mutate)
                    if callback:
                        with contextlib.suppress(Exception):
                            callback(job_id)
                    break

                if self.is_done(job):
                    break

                time.sleep(0.5)

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()

    def classify_scheduler_state(self: Self, state: Optional[str]) -> Optional[str]:
        """Map scheduler state to Huldra status."""
        if not state:
            return None

        s = state.upper()

        if "COMPLETE" in s or "COMPLETED" in s:
            return "success"

        if s in {
            "PREEMPTED",
            "TIMEOUT",
            "NODE_FAIL",
            "REQUEUED",
            "REQUEUE_HOLD",
        }:
            return "preempted"

        if s == "CANCELLED":
            return "preempted" if HULDRA_CONFIG.cancelled_is_preempted else "failed"

        if "FAIL" in s or "ERROR" in s:
            return "failed"

        return None

    def probe(self: Self, directory: Path, state: _HuldraState) -> dict[str, Any]:
        """
        Best-effort scheduler reconciliation.

        Returns a dict for `StateManager.reconcile(..., submitit_probe=...)`:
        - `terminal_status`: one of {failed, cancelled, preempted, crashed}
        - `scheduler_state`: raw scheduler state when available
        - `reason`: best-effort reason string
        """
        job = self.load_job(directory)
        if job is None:
            return {}

        scheduler_state = self.get_state(job)
        classified = self.classify_scheduler_state(scheduler_state)
        if classified is None:
            if self.is_done(job):
                return {
                    "terminal_status": "crashed",
                    "scheduler_state": scheduler_state,
                    "reason": "job_done_unknown_state",
                }
            return {}

        # `COMPLETED` doesn't guarantee the worker wrote a success marker/state.
        if classified == "success":
            return {
                "terminal_status": "crashed",
                "scheduler_state": scheduler_state,
                "reason": "scheduler_completed_no_success_marker",
            }

        return {
            "terminal_status": classified,
            "scheduler_state": scheduler_state,
            "reason": f"scheduler:{scheduler_state}",
        }
