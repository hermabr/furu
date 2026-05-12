from __future__ import annotations

from typing import Any

__all__ = ["ServerUnavailable", "worker_loop"]


def __getattr__(name: str) -> Any:
    if name == "ServerUnavailable":
        from furu.worker.loop import ServerUnavailable

        return ServerUnavailable
    if name == "worker_loop":
        from furu.worker.loop import worker_loop

        return worker_loop
    raise AttributeError(name)
