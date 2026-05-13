from __future__ import annotations

import threading
from typing import Protocol

from furu.worker.loop import worker_loop


class WorkerPool(Protocol):
    def start(self, *, server_url: str, token: str | None) -> None: ...

    def check(self) -> str | None: ...

    def stop(self) -> None: ...


class LocalWorkerPool:
    def __init__(self, *, n_workers: int, stop_timeout: float = 5.0) -> None:
        if n_workers < 1:
            raise ValueError("LocalWorkerPool requires at least one worker")
        self._n_workers = n_workers
        self._stop_timeout = stop_timeout
        self._workers: list[threading.Thread] = []

    def start(self, *, server_url: str, token: str | None) -> None:
        if self._workers:
            raise RuntimeError("worker pool has already been started")

        self._workers = [
            threading.Thread(
                target=worker_loop,
                kwargs={"server_url": server_url, "token": token},
                name=f"furu-worker-{idx}",
            )
            for idx in range(self._n_workers)
        ]

        for worker in self._workers:
            worker.start()

    def check(self) -> str | None:
        for worker in self._workers:
            if not worker.is_alive():
                return f"{worker.name} exited before manager run completed"
        return None

    def stop(self) -> None:
        for worker in self._workers:
            worker.join(timeout=self._stop_timeout)
