from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from furu.core import Spec
    from furu.resources import Worker
    from furu.worker.messages import Pool


class WorkerBackend(Protocol):
    """What a script passes to ``create(..., on=[...])``: a pool's recipe."""

    @property
    def worker(self) -> Worker:
        """What every worker in this pool offers."""
        ...

    def slices(self) -> list[tuple[int, Worker]]:
        """Each slot count a job can take on a worker, fewest first, and what
        it offers; the last is the whole worker."""
        ...

    @property
    def accepts(self) -> Callable[[Spec], bool] | None:
        """Which specs this pool's workers take; None takes every spec."""
        ...

    @property
    def pool_key(self) -> str:
        """Equal for backends whose workers are interchangeable."""
        ...

    def pool(self) -> Pool:
        """The plain data the manager needs to start this pool's workers."""
        ...


class WorkerPool(Protocol):
    """A manager's running pool."""

    def job_completed(self) -> None: ...

    def stop(self) -> None: ...


def slice_for(backend: WorkerBackend, spec: Spec) -> tuple[int, Worker] | None:
    """The fewest slots of ``backend``'s workers that ``spec`` runs on, and what
    they offer; None if no worker of it can run ``spec``."""
    if backend.accepts is not None and not backend.accepts(spec):
        return None
    return next(((k, w) for k, w in backend.slices() if spec.runs_on(w)), None)
