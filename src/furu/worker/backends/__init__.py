from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from furu.logging import get_logger

if TYPE_CHECKING:
    from furu.execution.api import PoolApiClient
    from furu.resources import ResourceRequest


logger = get_logger()


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
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def join(self, *, timeout: float) -> None: ...


class _SelfScalingWorkerPool:
    """Base for worker pools that drive their own scaling in a background thread.

    Subclasses implement `_scale_once` (one scaling iteration) and
    `_workers_healthy` (health check). The pool reports failures back to the
    manager via the manager API so it remains self-contained — no centralized
    process needs to poll the pool.
    """

    def __init__(
        self,
        *,
        client: PoolApiClient,
        scale_interval: float,
        description: str,
    ) -> None:
        self._client = client
        self._scale_interval = scale_interval
        self._description = description
        self._stop_event = threading.Event()
        self._scale_thread: threading.Thread | None = None

    def _scale_once(self) -> None:
        raise NotImplementedError

    def _workers_healthy(self) -> bool:
        raise NotImplementedError

    def start(self) -> None:
        if self._scale_thread is not None:
            raise RuntimeError(f"{self._description} already started")
        self._scale_once()
        self._scale_thread = threading.Thread(
            target=self._scale_loop,
            name=f"furu-{self._description}-scale",
        )
        self._scale_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _join_scale_loop(self, *, timeout: float) -> None:
        if self._scale_thread is not None:
            self._scale_thread.join(timeout=timeout)

    def _scale_loop(self) -> None:
        try:
            while not self._stop_event.wait(timeout=self._scale_interval):
                self._scale_once()
                if not self._workers_healthy():
                    self._report_failure(f"{self._description} became unhealthy")
                    return
        except Exception as exc:
            self._report_failure(
                f"{self._description} scale loop crashed: "
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )

    def _report_failure(self, message: str) -> None:
        logger.error("%s failure: %s", self._description, message)
        try:
            self._client.fail(message=message)
        except Exception:
            logger.exception(
                "failed to report %s failure to manager", self._description
            )


def count_workers_to_launch(
    client: PoolApiClient,
    *,
    current_workers: int,
    max_workers: int,
    resource_request: ResourceRequest,
) -> int:
    if current_workers >= max_workers:
        return 0
    return client.count_satisfiable_jobs(
        resources=resource_request,
        max_workers=max_workers - current_workers,
    )
