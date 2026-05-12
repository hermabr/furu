from __future__ import annotations

import socket
import threading
from collections.abc import Sequence

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from furu.logging import get_logger
from furu.manager import GetJobResponse, Manager
from furu.metadata import ArtifactSpec


class FinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    success: bool


class BlockedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    dependencies: list[ArtifactSpec]


def make_app(manager: Manager) -> FastAPI:
    app = FastAPI()

    @app.get("/get_job")
    def get_job() -> GetJobResponse:
        return manager.get_job()

    @app.post("/finish/{lease_id}")
    def finish(lease_id: str, body: FinishRequest) -> dict[str, bool]:
        manager.finish(lease_id, success=body.success)
        return {"ok": True}

    @app.post("/blocked/{lease_id}")
    def blocked(lease_id: str, body: BlockedRequest) -> dict[str, bool]:
        manager.report_blocked(lease_id, body.dependencies)
        return {"ok": True}

    return app


class _ServerHandle:
    """Background uvicorn server bound to an OS-assigned port."""

    def __init__(self, server: uvicorn.Server, thread: threading.Thread, port: int):
        self.server = server
        self.thread = thread
        self.port = port

    @classmethod
    def start(cls, app: FastAPI, host: str = "127.0.0.1") -> _ServerHandle:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        port = sock.getsockname()[1]

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            daemon=True,
            name="furu-manager-server",
        )
        thread.start()

        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("manager server thread exited before startup")
            threading.Event().wait(0.02)

        return cls(server=server, thread=thread, port=port)

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout)


def run_until_done(
    manager: Manager,
    *,
    n_workers: int,
    host: str = "127.0.0.1",
) -> None:
    from furu.worker import spawn_workers

    logger = get_logger("manager")
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    app = make_app(manager)
    handle = _ServerHandle.start(app, host=host)
    base_url = f"http://{host}:{handle.port}"
    logger.info(
        "manager server listening on %s; spawning %d worker(s)",
        base_url,
        n_workers,
    )

    worker_threads: Sequence[threading.Thread] = ()
    try:
        worker_threads = spawn_workers(base_url, n_workers=n_workers)
        for thread in worker_threads:
            thread.join()
    finally:
        handle.stop()

    manager.log_summary()
