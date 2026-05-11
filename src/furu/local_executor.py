from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from furu.execution import BlockedOnDependencies, _run_single_artifact
from furu.scheduler import Job, Scheduler

if TYPE_CHECKING:
    from furu.core import Furu


def run_local_executor(
    finals: Sequence[Furu[Any]],
    *,
    scheduler: Scheduler | None = None,
) -> Scheduler:
    """Run the given finals to completion using a single in-process worker.

    Builds a dependency graph from each final's declared references (fields
    plus ``@furu.dependency`` properties) and executes each artifact in
    topological order, discovering additional dynamic dependencies through
    suspension when an artifact's ``create()`` calls ``load_or_create()`` on
    something that is not yet computed.

    Returns the :class:`~furu.scheduler.Scheduler` so callers can inspect
    final job states. Raises ``RuntimeError`` if execution stalls (the
    graph is empty of runnable jobs but not yet done) or if any job ended
    up failed.
    """
    if scheduler is None:
        scheduler = Scheduler()
    scheduler.submit(finals)

    while not scheduler.is_done():
        job = scheduler.claim_ready()
        if job is None:
            raise RuntimeError(
                "local executor stalled: no runnable jobs but graph is not done"
            )
        _run_claimed_job(job, scheduler)

    if scheduler.has_failures():
        failures = [
            (job.obj._log_label, job.error)
            for job in scheduler.jobs().values()
            if job.state == "failed"
        ]
        first_label, first_exc = failures[0]
        raise RuntimeError(
            f"local executor finished with {len(failures)} failed job(s); "
            f"first failure was {first_label}"
        ) from first_exc

    return scheduler


def _run_claimed_job(job: Job, scheduler: Scheduler) -> None:
    obj = job.obj
    try:
        _run_single_artifact(obj)
    except BlockedOnDependencies as exc:
        if not scheduler.suspend_job(obj):
            return
        scheduler.add_dependencies(obj, exc.deps)
    except BaseException as exc:
        scheduler.fail_job(obj, exc)
    else:
        scheduler.complete_job(obj)
