from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from furu.execution.api import PoolApiClient
from furu.resources import ResourceRequest
from furu.worker.backends import count_workers_to_launch, run_pool_management_loop


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
        pool = LocalThreadWorkerPool(
            server_url=server_url,
            auth_token=auth_token,
            max_workers=self.max_workers,
            resource_request=self.resource_request,
        )
        pool.scale()
        return pool


class LocalThreadWorkerPool:
    management_interval = 0.1

    def __init__(
        self,
        *,
        server_url: str,
        auth_token: str,
        max_workers: int,
        resource_request: ResourceRequest,
    ) -> None:
        self._server_url = server_url
        self._auth_token = auth_token
        self._max_workers = max_workers
        self._resource_request = resource_request
        self._client = PoolApiClient(server_url, auth_token=auth_token)
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._management_thread: threading.Thread | None = None

    @property
    def n_workers(self) -> int:
        return len(self._threads)

    def scale(self) -> None:
        to_spawn = count_workers_to_launch(
            self._client,
            current_workers=len(self._threads),
            max_workers=self._max_workers,
            resource_request=self._resource_request,
        )
        from furu.worker.loop import worker_loop

        for _ in range(to_spawn):
            thread = threading.Thread(
                target=worker_loop,
                kwargs={
                    "server_url": self._server_url,
                    "auth_token": self._auth_token,
                    "resource_request": self._resource_request,
                },
                name=f"furu-worker-{len(self._threads)}",
            )
            self._threads.append(thread)
            thread.start()

    def is_healthy(self) -> bool:
        return all(worker.is_alive() for worker in self._threads)

    def start(self) -> None:
        if self._management_thread is not None:
            raise RuntimeError("LocalThreadWorkerPool already started")
        self._management_thread = threading.Thread(
            target=run_pool_management_loop,
            kwargs={
                "scale": self.scale,
                "is_healthy": self.is_healthy,
                "interval": self.management_interval,
                "stop_event": self._stop_event,
                "client": self._client,
                "pool_name": type(self).__name__,
            },
            name="furu-local-pool-manager",
        )
        self._management_thread.start()

    def join(self, *, timeout: float) -> None:
        self._stop_event.set()
        if self._management_thread is not None:
            self._management_thread.join(timeout=timeout)
        for worker in self._threads:
            worker.join(timeout=timeout)
