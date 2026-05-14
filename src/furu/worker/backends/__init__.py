from __future__ import annotations

from typing import Protocol


class WorkerBackend(Protocol):
    def start_pool(self, *, server_url: str, auth_token: str) -> WorkerPool: ...


class WorkerPool(Protocol):
    @property
    def health_check_interval(self) -> float: ...

    def is_healthy(self) -> bool: ...

    def join(self, *, timeout: float) -> None: ...
