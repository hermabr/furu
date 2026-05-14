from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from secrets import token_urlsafe

import uvicorn

from furu.execution.api import create_manager_api_app
from furu.execution.manager import Manager
from furu.worker.backends import WorkerBackend


@dataclass(frozen=True, slots=True)
class ManagerServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"http://{self.bound_host}:{self.bound_port}"


@contextmanager
def manager_server(
    manager: Manager, *, bind_host: str, port: int
) -> Iterator[ManagerServer]:
    auth_token = token_urlsafe(32)
    app = create_manager_api_app(manager, auth_token=auth_token)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None

    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, port))
        sock.listen()
        sock.set_inheritable(True)
        bound_host, bound_port = sock.getsockname()[:2]

        server = uvicorn.Server(
            uvicorn.Config(
                app,
                log_level="warning",
                lifespan="off",
                ws="none",
            )
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="furu-manager-server",
        )
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("manager server exited before it was ready")
            if time.monotonic() > deadline:
                raise TimeoutError("manager server did not start within 10 seconds")
            time.sleep(0.01)

        yield ManagerServer(
            bound_host=bound_host,
            bound_port=bound_port,
            auth_token=auth_token,
        )
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)
        sock.close()


def _run_until_done(
    manager: Manager,
    *,
    worker_backend: WorkerBackend,
    host: str,
    port: int,
) -> None:
    with manager_server(manager, bind_host=host, port=port) as server:
        worker_pool = worker_backend.start_pool(
            server_url=server.server_url,
            auth_token=server.auth_token,
            executor_dir=manager.executor_dir,
        )
        health_check_interval = worker_pool.health_check_interval

        while not manager.done.wait(timeout=health_check_interval):
            if not worker_pool.is_healthy():
                manager.fail(
                    "worker backend became unhealthy before manager run completed"
                )
                break

        worker_pool.join(timeout=5)

    manager.raise_for_failure()
