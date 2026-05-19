from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from furu.execution.api import ManagerApiClient
from furu.resources import ResourceRequest


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    n_workers: int = 1
    manager_listen_host: str = "127.0.0.1"

    def start_pool(
        self, *, server_url: str, auth_token: str, executor_dir: Path
    ) -> LocalThreadWorkerPool:
        n_workers = ManagerApiClient(
            server_url,
            auth_token=auth_token,
        ).count_satisfiable_ready_jobs(
            ResourceRequest(memory=0),
            max_workers=self.n_workers,
        )
        return LocalThreadWorkerPool(
            server_url=server_url,
            auth_token=auth_token,
            n_workers=n_workers,
        )


class LocalThreadWorkerPool:
    health_check_interval = 0.1

    def __init__(
        self,
        *,
        server_url: str,
        auth_token: str,
        n_workers: int,
    ) -> None:
        from furu.worker.loop import worker_loop

        self.n_workers = n_workers
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
