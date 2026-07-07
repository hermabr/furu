from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from furu.provenance import SubmitProvenance


class WorkerBackend(Protocol):
    execution_coordinator_listen_host: str

    def start_pool(
        self,
        *,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        provenance: SubmitProvenance,
    ) -> WorkerPool: ...


class WorkerPool(Protocol):
    def stop(self, *, timeout: float) -> None: ...
