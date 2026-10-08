from __future__ import annotations

import dataclasses
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from furu.config import get_config
from furu.logging import get_logger
from furu.resources import Worker
from furu.utils import _hash_dict_deterministically
from furu.worker.protocol import PoolHandoff, coordinator_url

if TYPE_CHECKING:
    from furu.core import Spec
    from furu.execution.execution_coordinator import ExecutionCoordinator


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    max_workers: int = 1
    worker: Worker = field(default_factory=Worker.here)
    accepts: Callable[[Spec], bool] | None = None
    execution_coordinator_listen_host: str = "127.0.0.1"

    @property
    def pool_key(self) -> str:
        return "local:" + _hash_dict_deterministically(
            {"worker": dataclasses.asdict(self.worker)}
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
                    "pool": self.pool_key,
                    "component": f"w{index}",
                },
                name=f"furu-local-worker-{index}",
            )
            threads.append(thread)
            thread.start()
        return LocalThreadWorkerPool(_threads=threads)


def _run_worker(
    *,
    coordinator: ExecutionCoordinator,
    coordinator_url: str,
    pool: str,
    component: str,
) -> None:
    from furu.worker.loop import worker_loop

    try:
        worker_loop(
            coordinator=coordinator_url,
            pool=pool,
            # Local threads are cheap to keep connected; they stay until the
            # server closes the connection.
            idle_timeout=None,
            max_failures=get_config().worker.max_failures_per_worker,
            component=component,
            backend="local-thread",
            materialize_snapshot=False,
        )
    except SystemExit as exc:  # gave up after too many failures; already logged
        coordinator.fail(str(exc.code))
    except Exception as exc:  # noqa: BLE001 -- fault barrier: fail the run instead
        get_logger(component).exception("local worker thread crashed")
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
