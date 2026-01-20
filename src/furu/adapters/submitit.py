import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol


from ..config import FURU_CONFIG
from ..storage import StateManager
from ..storage.state import _FuruState, ProbeResult


# Protocol for submitit Job-like objects. We use this instead of Any because
# submitit is an external library and we want to document the interface we expect.
class SubmititJobProtocol(Protocol):
    """Protocol for submitit Job objects."""

    job_id: str | None

    def done(self) -> bool: ...
    def state(self) -> str: ...
    def result(self, timeout: float | None = None) -> object: ...
    def wait(self) -> None: ...


# Type alias for submitit Executor. The executor is from an external library
# with a complex generic type, so we use Any here.
SubmititExecutor = Any

# Type alias for submitit Job. Jobs come from external library and can be
# various types depending on the executor backend.
SubmititJob = Any


class SubmititAdapter:
    """Adapter for working with submitit executors."""

    JOB_PICKLE = "job.pkl"

    def __init__(self, executor: SubmititExecutor):
        self.executor = executor

    def submit(self, fn: Callable[[], None]) -> SubmititJob:
        """Submit a job to the executor."""
        return self.executor.submit(fn)

    def wait(self, job: SubmititJob, timeout: float | None = None) -> None:
        """Wait for job completion."""
        if timeout:
            job.result(timeout=timeout)
        else:
            job.wait()

    def get_job_id(self, job: SubmititJob) -> str | None:
        """Get job ID if available."""
        job_id = getattr(job, "job_id", None)
        if job_id:
            return str(job_id)
        return None

    def is_done(self, job: SubmititJob) -> bool:
        """Check if job is done."""
        done_fn = getattr(job, "done", None)
        if done_fn and callable(done_fn):
            return done_fn()
        return False

    def get_state(self, job: SubmititJob) -> str | None:
        """Get job state from scheduler."""
        state_fn = getattr(job, "state", None)
        if state_fn and callable(state_fn):
            return state_fn()
        return None

    def pickle_job(self, job: SubmititJob, directory: Path) -> None:
        """Pickle job handle to file."""
        import cloudpickle as pickle

        job_path = StateManager.get_internal_dir(directory) / self.JOB_PICKLE
        job_path.parent.mkdir(parents=True, exist_ok=True)
        with job_path.open("wb") as f:
            pickle.dump(job, f)

    def load_job(self, directory: Path) -> SubmititJob | None:
        """Load job handle from pickle file."""
        job_path = StateManager.get_internal_dir(directory) / self.JOB_PICKLE
        if not job_path.is_file():
            return None

        import cloudpickle as pickle

        with job_path.open("rb") as f:
            return pickle.load(f)

    def watch_job_id(
        self,
        job: SubmititJob,
        directory: Path,
        *,
        attempt_id: str,
        callback: Callable[[str], None] | None = None,
    ) -> None:
        """Watch for job ID in background thread and update state."""

        def watcher():
            while True:
                job_id = self.get_job_id(job)
                if job_id:

                    def mutate(state: _FuruState) -> None:
                        attempt = state.attempt
                        if attempt is None or attempt.id != attempt_id:
                            return
                        attempt.scheduler["job_id"] = job_id

                    StateManager.update_state(directory, mutate)
                    if callback:
                        callback(job_id)
                    break

                if self.is_done(job):
                    break

                time.sleep(0.5)

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()

    def classify_scheduler_state(self, state: str | None) -> str | None:
        """Map scheduler state to Furu status."""
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
            return "preempted" if FURU_CONFIG.cancelled_is_preempted else "failed"

        if "FAIL" in s or "ERROR" in s:
            return "failed"

        return None

    def probe(self, directory: Path, state: _FuruState) -> ProbeResult:
        """
        Best-effort scheduler reconciliation.

        Returns a dict for `StateManager.reconcile(..., submitit_probe=...)`:
        - `terminal_status`: one of {failed, cancelled, preempted, crashed}
        - `scheduler_state`: raw scheduler state when available
        - `reason`: best-effort reason string

        Returns empty dict if job status cannot be determined (e.g., job pickle
        doesn't exist yet), allowing reconcile to fall back to lease expiry.
        """
        job = self.load_job(directory)
        if job is None:
            # Job pickle doesn't exist - can't determine status, fall back to lease expiry
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
