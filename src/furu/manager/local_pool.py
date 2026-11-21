from __future__ import annotations

import threading
import traceback
from collections.abc import Callable

from furu.logging import get_logger
from furu.protocol.messages import ErrorMessage, LocalPool, Pool
from furu.worker.loop import worker_loop


class LocalThreadPool:
    """``max_workers`` threads that connect to ``url`` and stay until it closes."""

    def __init__(
        self,
        *,
        url: str,
        key: str,
        definition: LocalPool,
        notify: Callable[[ErrorMessage], None],
    ) -> None:
        self.definition: Pool = definition
        self._threads = [
            threading.Thread(
                target=_run_worker,
                args=(url, key, f"w{index}", notify),
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


def _run_worker(
    url: str, key: str, component: str, notify: Callable[[ErrorMessage], None]
) -> None:
    try:
        worker_loop(
            url=url,
            pool=key,
            idle_timeout=None,
            component=component,
            backend="local-thread",
        )
    except Exception as exc:  # noqa: BLE001 -- fault barrier: fail the runs instead
        get_logger(component).exception("local worker thread crashed")
        notify(
            ErrorMessage(
                reason="local worker thread crashed: "
                + traceback.format_exception_only(type(exc), exc)[-1].strip()
            )
        )
