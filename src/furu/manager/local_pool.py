from __future__ import annotations

import threading
import traceback
from typing import TYPE_CHECKING

from furu.logging import get_logger
from furu.protocol.messages import ErrorMessage, LocalPool, Pool
from furu.worker.loop import worker_loop

if TYPE_CHECKING:
    from furu.manager.manager import Manager


class LocalThreadPool:
    """``max_workers`` threads that stay until the manager closes."""

    def __init__(self, *, manager: Manager, key: str, definition: LocalPool) -> None:
        self.definition: Pool = definition
        self._threads = [
            threading.Thread(
                target=_run_worker,
                args=(manager, key, f"w{index}"),
                name=f"furu-local-worker-{index}",
            )
            for index in range(definition.max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def job_completed(self) -> None:
        pass

    def stop(self) -> None:
        for thread in self._threads:
            thread.join(timeout=10)


def _run_worker(manager: Manager, key: str, component: str) -> None:
    try:
        worker_loop(
            url=manager.url,
            pool=key,
            idle_timeout=None,
            component=component,
            backend="local-thread",
        )
    except Exception as exc:  # noqa: BLE001 -- fault barrier: fail the runs instead
        get_logger(component).exception("local worker thread crashed")
        manager.notify(
            key,
            ErrorMessage(
                reason="local worker thread crashed: "
                + traceback.format_exception_only(type(exc), exc)[-1].strip()
            ),
        )
