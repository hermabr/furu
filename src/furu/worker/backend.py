from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkerPool(Protocol):
    def start(self) -> None: ...

    def first_dead_worker(self) -> str | None: ...

    def join(self, *, timeout: float) -> None: ...


@runtime_checkable
class WorkerBackend(Protocol):
    def create_pool(self, *, n_workers: int, server_url: str) -> WorkerPool: ...
