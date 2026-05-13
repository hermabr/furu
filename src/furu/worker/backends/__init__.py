from typing import Protocol


class WorkerPool(Protocol):
    def is_healthy(self) -> bool: ...

    def join(self, *, timeout: float | None = None) -> None: ...


class WorkerBackend(Protocol):
    def start_pool(self, *, server_url: str) -> WorkerPool: ...
