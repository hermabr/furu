from __future__ import annotations

import dataclasses
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from furu.logging import get_logger
from furu.resources import Worker
from furu.utils import _hash_dict_deterministically
from furu.worker.messages import ErrorMessage, LocalPool

if TYPE_CHECKING:
    from furu.core import Spec
    from furu.execution.manager import Manager


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    """Worker threads in this process, running the live worktree."""

    max_workers: int = 1
    worker: Worker = field(default_factory=Worker.here)
    accepts: Callable[[Spec], bool] | None = None

    @property
    def pool_key(self) -> str:
        return "local:" + _hash_dict_deterministically(
            {"worker": dataclasses.asdict(self.worker)}
        )

    def pool(self) -> LocalPool:
        return LocalPool(max_workers=self.max_workers)


class LocalThreadPool:
    """``max_workers`` threads that stay until the manager closes."""

    def __init__(self, *, manager: Manager, key: str, definition: LocalPool) -> None:
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
    from furu.worker.loop import worker_loop

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
