from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from furu.core import Spec
    from furu.resources import Worker
    from furu.worker.messages import Pool


class WorkerBackend(Protocol):
    """What a script passes to ``create(..., on=[...])``: a pool's recipe."""

    @property
    def workers(self) -> Mapping[int, Worker]:
        """What a job gets in this pool for each number of slots it may take,
        fewest first. The last is a whole allocation."""
        ...

    @property
    def accepts(self) -> Callable[[Spec], bool] | None:
        """Which specs this pool's workers take; None takes every spec."""
        ...

    @property
    def pool_key(self) -> str:
        """Equal for backends whose allocations are interchangeable."""
        ...

    def pool(self) -> Pool:
        """The plain data the manager needs to start this pool's allocations."""
        ...


class WorkerPool(Protocol):
    """A manager's running pool."""

    definition: Pool

    def job_completed(self) -> None: ...

    def stop(self) -> None: ...


def slice_for(backend: WorkerBackend, spec: Spec) -> tuple[int, Worker] | None:
    """The fewest slots of ``backend`` that run ``spec``, and what they offer."""
    if backend.accepts is not None and not backend.accepts(spec):
        return None
    return next(
        (
            (slots, worker)
            for slots, worker in backend.workers.items()
            if spec.runs_on(worker)
        ),
        None,
    )
