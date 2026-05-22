from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from furu.execution.api import PoolApiClient
from furu.logging import get_logger
from furu.resources import ResourceRequest
from furu.worker.backends import _SelfScalingWorkerPool, count_workers_to_launch

logger = get_logger()


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    max_workers: int = 1
    resource_request: ResourceRequest = field(
        default_factory=lambda: ResourceRequest(memory=sys.maxsize)
    )
    manager_listen_host: str = "127.0.0.1"
    scale_interval: float = 0.1

    def start_pool(
        self,
        *,
        server_url: str,
        auth_token: str,
        executor_dir: Path,
    ) -> LocalThreadWorkerPool:
        return LocalThreadWorkerPool(
            server_url=server_url,
            auth_token=auth_token,
            max_workers=self.max_workers,
            resource_request=self.resource_request,
            scale_interval=self.scale_interval,
        )


class LocalThreadWorkerPool(_SelfScalingWorkerPool):
    def __init__(
        self,
        *,
        server_url: str,
        auth_token: str,
        max_workers: int,
        resource_request: ResourceRequest,
        scale_interval: float = 0.1,
    ) -> None:
        super().__init__(
            client=PoolApiClient(server_url=server_url, auth_token=auth_token),
            scale_interval=scale_interval,
            description="local worker pool",
        )
        self._server_url = server_url
        self._auth_token = auth_token
        self._max_workers = max_workers
        self._resource_request = resource_request
        self._threads: list[threading.Thread] = []
        self._crashed_workers: list[BaseException] = []
        self._crash_lock = threading.Lock()

    @property
    def n_workers(self) -> int:
        return len(self._threads)

    def _stop_workers(self, *, timeout: float) -> None:
        for worker in self._threads:
            worker.join(timeout=timeout)

    def _scale_once(self) -> None:
        to_spawn = count_workers_to_launch(
            self._client,
            current_workers=len(self._threads),
            max_workers=self._max_workers,
            resource_request=self._resource_request,
        )
        for _ in range(to_spawn):
            thread = threading.Thread(
                target=self._run_worker,
                name=f"furu-worker-{len(self._threads)}",
            )
            self._threads.append(thread)
            thread.start()

    def _run_worker(self) -> None:
        from furu.worker.loop import worker_loop

        try:
            worker_loop(
                server_url=self._server_url,
                auth_token=self._auth_token,
                resource_request=self._resource_request,
            )
        except BaseException as exc:
            with self._crash_lock:
                self._crashed_workers.append(exc)
            logger.exception("local worker thread crashed")

    def _workers_healthy(self) -> bool:
        with self._crash_lock:
            return not self._crashed_workers
