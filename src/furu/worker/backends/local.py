from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    n_workers: int = 1

    def __post_init__(self) -> None:
        if self.n_workers < 1:
            raise ValueError("LocalThreadWorkerBackend requires at least one worker")

    def create_pool(self, *, server_url: str) -> LocalThreadWorkerPool:
        return LocalThreadWorkerPool(server_url=server_url, n_workers=self.n_workers)


class LocalThreadWorkerPool:
    def __init__(self, *, server_url: str, n_workers: int) -> None:
        if n_workers < 1:
            raise ValueError("LocalThreadWorkerPool requires at least one worker")

        from furu.worker.loop import worker_loop

        self._threads = [
            threading.Thread(
                target=worker_loop,
                kwargs={"server_url": server_url},
                name=f"furu-worker-{idx}",
            )
            for idx in range(n_workers)
        ]
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("LocalThreadWorkerPool has already been started")

        self._started = True
        for worker in self._threads:
            worker.start()

    def is_healthy(self) -> bool:
        return not self._started or all(worker.is_alive() for worker in self._threads)

    def join(self, *, timeout: float | None = None) -> None:
        for worker in self._threads:
            worker.join(timeout=timeout)
