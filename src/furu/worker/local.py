from __future__ import annotations

import threading
from collections.abc import Sequence

from furu.worker.backend import WorkerPool
from furu.worker.loop import worker_loop


class LocalThreadWorkerPool:
    def __init__(self, *, n_workers: int, server_url: str) -> None:
        self._workers: Sequence[threading.Thread] = [
            threading.Thread(
                target=worker_loop,
                kwargs={"server_url": server_url},
                name=f"furu-worker-{idx}",
            )
            for idx in range(n_workers)
        ]

    def start(self) -> None:
        for worker in self._workers:
            worker.start()

    def first_dead_worker(self) -> str | None:
        for worker in self._workers:
            if not worker.is_alive():
                return worker.name
        return None

    def join(self, *, timeout: float) -> None:
        for worker in self._workers:
            worker.join(timeout=timeout)


class LocalThreadBackend:
    def create_pool(self, *, n_workers: int, server_url: str) -> WorkerPool:
        return LocalThreadWorkerPool(n_workers=n_workers, server_url=server_url)
