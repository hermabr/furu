from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from furu.provenance import SubmitProvenance
    from furu.resources import ResourceRequest


class PoolCoordinator(Protocol):
    """The slice of the execution coordinator that worker pools call directly.

    Pools run in the coordinator's process; only workers talk over the wire.
    """

    def count_satisfiable_jobs(
        self, *, resources: ResourceRequest, max_workers: int
    ) -> int: ...

    def fail(self, message: str) -> None: ...


class WorkerBackend(Protocol):
    @property
    def execution_coordinator_listen_host(self) -> str: ...

    def start_pool(
        self,
        *,
        coordinator: PoolCoordinator,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        provenance: SubmitProvenance,
    ) -> WorkerPool: ...


class WorkerPool(Protocol):
    def stop(self, *, timeout: float) -> None: ...
