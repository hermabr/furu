from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from furu.execution.api import ManagerApiClient
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
        return LocalThreadWorkerPool(
            server_url=server_url,
            auth_token=auth_token,
            max_workers=self.max_workers,
            resource_request=self.resource_request,
        )


class LocalThreadWorkerPool:
    health_check_interval = 0.1

    def __init__(
        self,
        *,
        server_url: str,
        auth_token: str,
        max_workers: int,
        resource_request: ResourceRequest,
    ) -> None:
        from furu.worker.loop import worker_loop

        self._server_url = server_url
        self._auth_token = auth_token
        self._max_workers = max_workers
        self._resource_request = resource_request
        self._worker_loop = worker_loop
        self._client = ManagerApiClient(server_url, auth_token=auth_token)
        self._threads: list[threading.Thread] = []
        self._start_available_workers()

    @property
    def n_workers(self) -> int:
        return len(self._threads)

    def _start_available_workers(self) -> None:
        available_slots = self._max_workers - len(self._threads)
        if available_slots <= 0:
            return

        n_workers = self._client.count_satisfiable_jobs(
            resources=self._resource_request,
            max_workers=available_slots,
        )
        n_workers = max(0, min(n_workers, available_slots))
        start_idx = len(self._threads)
        threads = [
            threading.Thread(
                target=self._worker_loop,
                kwargs={
                    "server_url": self._server_url,
                    "auth_token": self._auth_token,
                    "resource_request": self._resource_request,
                },
                name=f"furu-worker-{idx}",
            )
            for idx in range(start_idx, start_idx + n_workers)
        ]
        self._threads.extend(threads)
        for worker in threads:
            worker.start()

    def is_healthy(self) -> bool:
        if not all(worker.is_alive() for worker in self._threads):
            return False
        self._start_available_workers()
        return all(worker.is_alive() for worker in self._threads)

    def join(self, *, timeout: float) -> None:
        for worker in self._threads:
            worker.join(timeout=timeout)
