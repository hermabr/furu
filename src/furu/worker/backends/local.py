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
        self._scale_thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self._crashed_workers: list[BaseException] = []
        self._crash_lock = threading.Lock()

    @property
    def n_workers(self) -> int:
        return len(self._threads)

    def start(self) -> None:
        if self._scale_thread is not None:
            raise RuntimeError("local worker pool already started")

        self._scale_once()
        self._scale_thread = threading.Thread(
            target=self._scale_loop,
            name="furu-local-worker-pool-scale",
        )
        self._scale_thread.start()

    def stop(self, *, timeout: float) -> None:
        self._stop_event.set()
        if self._scale_thread is not None:
            self._scale_thread.join(timeout=timeout)
        for worker in self._threads:
            worker.join(timeout=timeout)

    def _scale_once(self) -> None:
        if len(self._threads) >= self._max_workers:
            return

        to_spawn = self._client.count_satisfiable_jobs(
            resources=self._resource_request,
            max_workers=self._max_workers - len(self._threads),
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

    def _scale_loop(self) -> None:
        try:
            while not self._stop_event.wait(timeout=self._scale_interval):
                self._scale_once()
                if not self._workers_healthy():
                    self._report_failure("local worker pool became unhealthy")
                    return
        except Exception as exc:
            self._report_failure(
                "local worker pool scale loop crashed: "
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )

    def _report_failure(self, message: str) -> None:
        logger.error("local worker pool failure: %s", message)
        try:
            self._client.fail(message=message)
        except Exception:
            logger.exception("failed to report local worker pool failure to manager")
