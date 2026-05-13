from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import socket
import threading
import time

import uvicorn

from furu.execution.api import create_manager_api_app
from furu.execution.manager import Manager
from furu.worker.backends import WorkerBackend


@dataclass(frozen=True, slots=True)
class ManagerServer:
    bound_host: str
    bound_port: int

    def url_for_workers(self, advertise_host: str | None = None) -> str:
        host = advertise_host if advertise_host is not None else self.bound_host
        return f"http://{host}:{self.bound_port}"


@contextmanager
def manager_server(
    manager: Manager,
    *,
    bind_host: str,
    port: int,
    startup_timeout: float = 10.0,
) -> Iterator[ManagerServer]:
    app = create_manager_api_app(manager)
    server: uvicorn.Server | None = None
    server_thread: threading.Thread | None = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

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
        server_thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="furu-manager-server",
        )
        server_thread.start()
        _wait_until_started(server, server_thread, startup_timeout)

        yield ManagerServer(bound_host=bound_host, bound_port=bound_port)
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None and server_thread.ident is not None:
            server_thread.join(timeout=10)
        sock.close()


def _wait_until_started(
    server: uvicorn.Server,
    server_thread: threading.Thread,
    startup_timeout: float,
) -> None:
    deadline = time.monotonic() + startup_timeout
    while not server.started:
        if not server_thread.is_alive():
            raise RuntimeError("manager server exited before it was ready")
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"manager server did not start within {startup_timeout:g} seconds"
            )
        time.sleep(0.01)


def _run_until_done(
    manager: Manager,
    *,
    worker_backend: WorkerBackend,
    host: str,
    port: int,
) -> None:
    with manager_server(manager, bind_host=host, port=port) as server:
        worker_pool = worker_backend.start_pool(server_url=server.url_for_workers())
        while not manager.done.wait(timeout=0.1):
            if not worker_pool.is_healthy():
                manager.fail(
                    "worker backend became unhealthy before manager run completed"
                )
                break

        worker_pool.join(timeout=5)

    manager.raise_for_failure()
