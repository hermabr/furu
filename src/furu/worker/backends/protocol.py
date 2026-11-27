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
        """What each allocation in this pool offers, all its slots together."""
        ...

    @property
    def slots(self) -> int:
        """The even shares of an allocation that a job takes one or more of."""
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

    definition: Pool

    def job_completed(self) -> None: ...

    def stop(self) -> None: ...


def slots_for(backend: WorkerBackend, spec: Spec) -> int | None:
    """The fewest slots of ``backend``'s allocations that run ``spec``, if any do."""
    if backend.accepts is not None and not backend.accepts(spec):
        return None
    return next(
        (
            k
            for k in range(1, backend.slots + 1)
            if spec.runs_on(backend.worker.share(k, backend.slots))
        ),
        None,
    )
