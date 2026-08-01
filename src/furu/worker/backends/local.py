from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from furu.logging import _scoped_component, get_logger
from furu.provenance import SubmitProvenance
from furu.resources import ResourceRequest

if TYPE_CHECKING:
    from furu.execution.execution_coordinator import ExecutionCoordinator

logger = get_logger()


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    max_workers: int = 1
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    execution_coordinator_listen_host: str = "127.0.0.1"

    def start_pool(
        self,
        *,
        coordinator: ExecutionCoordinator,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        provenance: SubmitProvenance,
    ) -> LocalThreadWorkerPool:
        # Workers are threads in the submitting process, so they already run
        # the exact code the snapshot captured; ``provenance`` is unused.
        server_url = f"ws://{self.execution_coordinator_listen_host}:{bound_port}"
        threads = []
        for index in range(self.max_workers):
            thread = threading.Thread(
                target=_run_worker,
                kwargs={
                    "coordinator": coordinator,
                    "server_url": server_url,
                    "auth_token": auth_token,
                    "resource_request": self.resource_request,
                    "index": index,
                },
                name=f"local-worker-{index}",
            )
            threads.append(thread)
            thread.start()
        return LocalThreadWorkerPool(_threads=threads)


def _run_worker(
    *,
    coordinator: ExecutionCoordinator,
    server_url: str,
    auth_token: str,
    resource_request: ResourceRequest,
    index: int,
) -> None:
    from furu.worker.loop import worker_loop

    component = f"local-worker-{index}"
    try:
        worker_loop(
            server_url=server_url,
            auth_token=auth_token,
            resource_request=resource_request,
            # Local threads are cheap to keep connected; they stay until the
            # server closes the connection.
            idle_timeout=None,
            component=component,
            backend="local-thread",
        )
    except Exception as exc:
        with _scoped_component(component):
            logger.exception("local worker thread crashed")
        coordinator.fail(
            "local worker thread crashed: "
            + traceback.format_exception_only(type(exc), exc)[-1].strip()
        )


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerPool:
    _threads: list[threading.Thread]

    def stop(self, *, timeout: float) -> None:
        for thread in self._threads:
            thread.join(timeout=timeout)
