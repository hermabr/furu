from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from furu.execution.api import ManagerApiClient
    from furu.resources import ResourceRequest


class WorkerBackend(Protocol):
    manager_listen_host: str

    def start_pool(
        self,
        *,
        server_url: str,
        auth_token: str,
        executor_dir: Path,
    ) -> WorkerPool: ...


class WorkerPool(Protocol):
    @property
    def health_check_interval(self) -> float: ...

    def scale(self) -> None: ...

    def is_healthy(self) -> bool: ...

    def join(self, *, timeout: float) -> None: ...


def count_workers_to_launch(
    client: ManagerApiClient,
    *,
    current_workers: int,
    max_workers: int,
    resource_request: ResourceRequest,
) -> int:
    if current_workers >= max_workers:
        return 0
    return client.count_satisfiable_jobs(
        resources=resource_request,
        max_workers=max_workers - current_workers,
    )
