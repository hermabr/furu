from __future__ import annotations

from typing import Protocol


class WorkerBackend(Protocol):
    def start_pool(self, *, server_url: str) -> WorkerPool: ...


class WorkerPool(Protocol):
    def is_healthy(self) -> bool: ...

    def join(self, *, timeout: float) -> None: ...
