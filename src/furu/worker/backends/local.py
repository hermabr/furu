from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from furu.config import get_config
from furu.logging import _scoped_component, get_logger
from furu.resources import ResourceRequest, resource_request_adapter
from furu.utils import _hash_dict_deterministically
from furu.worker.protocol import PoolHandoff, coordinator_url

if TYPE_CHECKING:
    from furu.execution.execution_coordinator import ExecutionCoordinator

logger = get_logger()


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    max_workers: int = 1
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    execution_coordinator_listen_host: str = "127.0.0.1"

    @property
    def pool_key(self) -> str:
        return "local:" + _hash_dict_deterministically(
            {
                "resources": resource_request_adapter.dump_python(
                    self.resource_request, mode="json"
                ),
            }
        )

    def start_pool(
        self,
        *,
        coordinator: ExecutionCoordinator,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        handoff: PoolHandoff,
    ) -> LocalThreadWorkerPool:
        url = coordinator_url(
            host=self.execution_coordinator_listen_host,
            port=bound_port,
            auth_token=auth_token,
        )
        threads = []
        for index in range(self.max_workers):
            thread = threading.Thread(
                target=_run_worker,
                kwargs={
                    "coordinator": coordinator,
                    "coordinator_url": url,
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
    coordinator_url: str,
    resource_request: ResourceRequest,
    index: int,
) -> None:
    from furu.worker.loop import worker_loop

    component = f"local-worker-{index}"
    try:
        worker_loop(
            coordinator=coordinator_url,
            resource_request=resource_request,
            # Local threads are cheap to keep connected; they stay until the
            # server closes the connection.
            idle_timeout=None,
            max_failures=get_config().worker.max_failures_per_worker,
            component=component,
            backend="local-thread",
            materialize_snapshot=False,
        )
    except (Exception, SystemExit) as exc:
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

    def handoff(self) -> PoolHandoff:
        return PoolHandoff()
