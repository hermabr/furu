from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from furu.resources import ResourceRequest


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    max_workers: int = 1
    resource_request: ResourceRequest = field(
        default_factory=lambda: ResourceRequest(memory=sys.maxsize)
    )
    manager_listen_host: str = "127.0.0.1"

    def start_pool(
        self,
        *,
        server_url: str,
        auth_token: str,
        executor_dir: Path,
    ) -> LocalThreadWorkerPool:
        from furu.execution.api import ManagerApiClient

        client = ManagerApiClient(server_url, auth_token=auth_token)
        n_workers = client.count_satisfiable_jobs(
            resources=self.resource_request, max_workers=self.max_workers
        )
        return LocalThreadWorkerPool(
            server_url=server_url,
            auth_token=auth_token,
            n_workers=n_workers,
        )


class LocalThreadWorkerPool:
    health_check_interval = 0.1

    def __init__(self, *, server_url: str, auth_token: str, n_workers: int) -> None:
        from furu.worker.loop import worker_loop

        self._threads = [
            threading.Thread(
                target=worker_loop,
                kwargs={"server_url": server_url, "auth_token": auth_token},
                name=f"furu-worker-{idx}",
            )
            for idx in range(n_workers)
        ]
        for worker in self._threads:
            worker.start()

    def is_healthy(self) -> bool:
        return all(worker.is_alive() for worker in self._threads)

    def join(self, *, timeout: float) -> None:
        for worker in self._threads:
            worker.join(timeout=timeout)
