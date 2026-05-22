from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from furu.logging import get_logger

if TYPE_CHECKING:
    from furu.execution.api import PoolApiClient
    from furu.resources import ResourceRequest


logger = get_logger()


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
    def start(self) -> None: ...

    def join(self, *, timeout: float) -> None: ...


def count_workers_to_launch(
    client: PoolApiClient,
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


def run_pool_management_loop(
    *,
    scale: Callable[[], None],
    is_healthy: Callable[[], bool],
    interval: float,
    stop_event: threading.Event,
    client: PoolApiClient,
    pool_name: str,
) -> None:
    while not stop_event.is_set():
        try:
            scale()
            healthy = is_healthy()
        except Exception as exc:
            logger.exception("worker pool %s management loop errored", pool_name)
            _report_unhealthy(client, f"{pool_name} management loop errored: {exc!r}")
            return
        if not healthy:
            _report_unhealthy(client, f"{pool_name} became unhealthy")
            return
        stop_event.wait(interval)


def _report_unhealthy(client: PoolApiClient, reason: str) -> None:
    try:
        client.report_unhealthy(reason=reason)
    except Exception:
        logger.exception("failed to report pool unhealthy to manager: %s", reason)
