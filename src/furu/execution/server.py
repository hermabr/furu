from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast
from secrets import token_urlsafe

import uvicorn

from furu.execution.api import create_manager_api_app
from furu.execution.connection import ManagerConnection
from furu.execution.manager import Manager
from furu.logging import get_logger
from furu.worker.backends import WorkerBackend, WorkerPool

logger = get_logger()


@dataclass(frozen=True, slots=True)
class ManagerServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"http://{self.bound_host}:{self.bound_port}"

    @property
    def local_origin_url(self) -> str:
        host = self.bound_host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{self.bound_port}"


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
    worker_backends: tuple[WorkerBackend, ...],
    port: int,
    manager_connection: ManagerConnection | None = None,
) -> None:
    (bind_host,) = {backend.manager_listen_host for backend in worker_backends}
    connection = (
        manager_connection
        if manager_connection is not None
        else _select_manager_connection(worker_backends)
    )

    with manager.log_context():
        logger.info(
            "starting furu manager: executor_id=%s executor_dir=%s ready=%d blocked=%d",
            manager.executor_id,
            manager.executor_dir,
            len(manager.ready),
            len(manager.blocked),
        )
        with manager_server(manager, bind_host=bind_host, port=port) as server:
            with connection.connect(
                local_url=server.local_origin_url
            ) as advertised_url:
                logger.info(
                    "manager server listening: local_url=%s advertised_url=%s",
                    server.local_origin_url,
                    advertised_url,
                )
                pools: list[WorkerPool] = []
                for backend in worker_backends:
                    pool = backend.start_pool(
                        server_url=advertised_url,
                        auth_token=server.auth_token,
                        executor_dir=manager.executor_dir,
                    )
                    pools.append(pool)
                    logger.info(
                        "worker pool started: backend=%s", type(backend).__name__
                    )
                manager.done.wait()

                with ThreadPoolExecutor(max_workers=len(pools)) as executor:
                    for pool in pools:
                        executor.submit(pool.stop, timeout=5)
    manager.raise_for_failure()


def _select_manager_connection(
    worker_backends: tuple[WorkerBackend, ...],
) -> ManagerConnection:
    from furu.execution.connection import DirectManagerConnection

    connections: list[ManagerConnection] = []
    for backend in worker_backends:
        get_connection = cast(
            Callable[[], ManagerConnection | None] | None,
            getattr(backend, "manager_connection", None),
        )
        if get_connection is None:
            continue
        connection = get_connection()
        if connection is not None:
            connections.append(connection)

    if not connections:
        return DirectManagerConnection()

    first = connections[0]
    if any(connection != first for connection in connections[1:]):
        raise ValueError("worker backends requested conflicting manager connections")
    return first
