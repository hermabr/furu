from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from furu.config import get_config
from furu.execution.api import PoolApiClient
from furu.logging import get_logger
from furu.resources import ResourceRequest

logger = get_logger()


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    max_workers: int = 1
    max_failed_restarts: int = field(
        default_factory=lambda: get_config().worker.max_failed_restarts
    )
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    execution_coordinator_listen_host: str = "127.0.0.1"
    scale_interval: float = 1.0
    worker_idle_timeout: float = field(
        default_factory=lambda: get_config().worker.idle_timeout_seconds
    )

    def start_pool(
        self,
        *,
        server_url: str,
        auth_token: str,
        executor_dir: Path,
    ) -> LocalThreadWorkerPool:
        pool_holder: list[LocalThreadWorkerPool] = []
        pool = LocalThreadWorkerPool(
            _server_url=server_url,
            _auth_token=auth_token,
            _max_workers=self.max_workers,
            _max_failed_restarts=self.max_failed_restarts,
            _resource_request=self.resource_request,
            _scale_interval=self.scale_interval,
            _worker_idle_timeout=self.worker_idle_timeout,
            _client=PoolApiClient(server_url=server_url, auth_token=auth_token),
            _stop_event=threading.Event(),
            _unhealthy_event=threading.Event(),
            _scale_thread=threading.Thread(
                target=lambda: pool_holder[0]._scale_loop(),
                name="furu-local-worker-pool-scale",
            ),
            _threads=[],
            _failed_threads=[],
        )
        pool_holder.append(pool)
        pool._scale_thread.start()
        return pool


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerPool:
    _server_url: str
    _auth_token: str
    _max_workers: int
    _max_failed_restarts: int
    _resource_request: ResourceRequest
    _scale_interval: float
    _worker_idle_timeout: float
    _client: PoolApiClient
    _stop_event: threading.Event
    _unhealthy_event: threading.Event
    _scale_thread: threading.Thread
    _threads: list[threading.Thread]
    _failed_threads: list[threading.Thread]

    def stop(self, *, timeout: float) -> None:
        self._stop_event.set()
        self._scale_thread.join(timeout=timeout)
        for worker in self._threads:
            worker.join(timeout=timeout)

    def _scale_once(self) -> None:
        self._threads[:] = [thread for thread in self._threads if thread.is_alive()]
        remaining_starts = (
            self._max_workers
            + self._max_failed_restarts
            - len(self._failed_threads)
            - len(self._threads)
        )
        if len(self._threads) >= self._max_workers or remaining_starts <= 0:
            return

        to_spawn = self._client.count_satisfiable_jobs(
            resources=self._resource_request,
            max_workers=min(self._max_workers - len(self._threads), remaining_starts),
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
                idle_timeout=self._worker_idle_timeout,
            )
        except Exception:
            self._failed_threads.append(threading.current_thread())
            if len(self._failed_threads) > self._max_failed_restarts:
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
