from __future__ import annotations

from pathlib import Path
from typing import Protocol


class WorkerBackend(Protocol):
    manager_listen_host: str

    def start_pool(
        self,
        *,
        server_url: str,
        auth_token: str,
        executor_dir: Path,
    ) -> WorkerPool: ...


class WorkerPool(Protocol):
    def stop(self, *, timeout: float) -> None: ...
