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


def can_run(backend: WorkerBackend, spec: Spec) -> bool:
    return (backend.accepts is None or backend.accepts(spec)) and spec.runs_on(
        backend.worker
    )
