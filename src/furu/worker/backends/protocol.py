from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from furu.core import Spec
    from furu.execution.execution_coordinator import ExecutionCoordinator
    from furu.resources import Worker
    from furu.worker.protocol import PoolHandoff


class WorkerBackend(Protocol):
    @property
    def execution_coordinator_listen_host(self) -> str: ...

    @property
    def worker(self) -> Worker:
        """What every worker in this pool offers."""
        ...

    @property
    def accepts(self) -> Callable[[Spec], bool] | None:
        """Which specs this pool's workers take; None takes every spec."""
        ...

    @property
    def pool_key(self) -> str: ...

    def start_pool(
        self,
        *,
        coordinator: ExecutionCoordinator,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        handoff: PoolHandoff,
    ) -> WorkerPool: ...


class WorkerPool(Protocol):
    def stop(self, *, timeout: float) -> None: ...

    def handoff(self) -> PoolHandoff: ...


def can_run(backend: WorkerBackend, spec: Spec) -> bool:
    return (backend.accepts is None or backend.accepts(spec)) and spec.runs_on(
        backend.worker
    )
