from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from furu.execution.api import PoolApiClient
from furu.logging import get_logger
from furu.resources import ResourceRequest

logger = get_logger()


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    max_workers: int = 1
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    manager_listen_host: str = "127.0.0.1"
    scale_interval: float = 1.0

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


class LocalThreadWorkerPool:
    def __init__(
        self,
        *,
        server_url: str,
        auth_token: str,
        max_workers: int,
        resource_request: ResourceRequest,
        scale_interval: float,
    ) -> None:
        self._client = PoolApiClient(server_url=server_url, auth_token=auth_token)
        self._server_url = server_url
        self._auth_token = auth_token
        self._max_workers = max_workers
        self._resource_request = resource_request
        self._scale_interval = scale_interval
        self._stop_event = threading.Event()
        self._unhealthy_event = threading.Event()
        self._scale_thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._scale_thread is not None:
            raise RuntimeError("local worker pool already started")

        self._scale_thread = threading.Thread(
            target=self._scale_loop,
            name="furu-local-worker-pool-scale",
        )
        self._scale_thread.start()

    def stop(self, *, timeout: float) -> None:
        self._stop_event.set()
        if self._scale_thread is None:
            raise RuntimeError("local worker pool stop called before start")
        self._scale_thread.join(timeout=timeout)
        for worker in self._threads:
            worker.join(timeout=timeout)

    def _scale_once(self) -> None:
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        if len(self._threads) >= self._max_workers:
            return

        to_spawn = self._client.count_satisfiable_jobs(
            resources=self._resource_request,
            max_workers=self._max_workers - len(self._threads),
        )
        for _ in range(to_spawn):
            thread = threading.Thread(target=self._run_worker)
            thread.name = f"furu-worker-{id(thread)}"
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
        except Exception:
            self._unhealthy_event.set()
            logger.exception("local worker thread crashed")

    def _scale_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._scale_once()

                if self._unhealthy_event.is_set():
                    self._client.fail(message="local worker pool became unhealthy")
                    return

                if self._stop_event.wait(timeout=self._scale_interval):
                    return

        except Exception as exc:
            self._client.fail(
                message="local worker pool scale loop crashed: "
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )
