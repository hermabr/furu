from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from furu.execution.execution_coordinator import ExecutionCoordinator
    from furu.resources import ResourceRequest
    from furu.worker.protocol import PoolHandoff


class WorkerBackend(Protocol):
    @property
    def execution_coordinator_listen_host(self) -> str: ...

    @property
    def resource_request(self) -> ResourceRequest:
        """The resources presented by every worker in this pool."""
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
