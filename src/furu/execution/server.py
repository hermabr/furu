from __future__ import annotations

import secrets
import socket
import threading
import time
from types import TracebackType
from typing import Self

import uvicorn

from furu.execution.api import create_manager_api_app
from furu.execution.manager import Manager
from furu.execution.pool import LocalWorkerPool, WorkerPool


class ManagerServer:
    def __init__(
        self,
        manager: Manager,
        *,
        bind_host: str,
        port: int = 0,
        startup_timeout: float = 10.0,
        shutdown_timeout: float = 10.0,
        token: str | None = None,
    ) -> None:
        self._manager = manager
        self._bind_host = bind_host
        self._port = port
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self.token = secrets.token_urlsafe(32) if token is None else token

        self._bound_host: str | None = None
        self._bound_port: int | None = None
        self._server: uvicorn.Server | None = None
        self._server_thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        if self._server_thread is not None:
            raise RuntimeError("manager server has already been started")

        app = create_manager_api_app(self._manager, token=self.token)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self._bind_host, self._port))
            sock.listen()
            sock.set_inheritable(True)
        except BaseException:
            sock.close()
            raise

        bound_host, bound_port = sock.getsockname()[:2]
        self._bound_host = bound_host
        self._bound_port = bound_port
        self._socket = sock
        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                log_level="warning",
                lifespan="off",
                ws="none",
            )
        )
        self._server_thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [sock]},
            name="furu-manager-server",
        )

        try:
            self._server_thread.start()
            self._wait_until_started()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

        if self._server_thread is not None:
            self._server_thread.join(timeout=self._shutdown_timeout)
            if self._server_thread.is_alive():
                raise TimeoutError("manager server did not stop within timeout")

        if self._socket is not None:
            self._socket.close()

        self._server = None
        self._server_thread = None
        self._socket = None

    def url_for_workers(self, *, advertise_host: str | None = None) -> str:
        if self._bound_host is None or self._bound_port is None:
            raise RuntimeError("manager server has not been started")
        host = self._bound_host if advertise_host is None else advertise_host
        return f"http://{host}:{self._bound_port}"

    def _wait_until_started(self) -> None:
        if self._server is None or self._server_thread is None:
            raise RuntimeError("manager server has not been started")

        deadline = time.monotonic() + self._startup_timeout
        while not self._server.started:
            if not self._server_thread.is_alive():
                raise RuntimeError("manager server exited before it was ready")
            if time.monotonic() > deadline:
                raise TimeoutError("manager server did not start within timeout")
            time.sleep(0.01)


def _run_until_done(
    manager: Manager,
    *,
    n_workers: int,
    bind_host: str,
    port: int,
    advertise_host: str | None,
    pool: WorkerPool | None = None,
) -> None:
    worker_pool = LocalWorkerPool(n_workers=n_workers) if pool is None else pool

    with ManagerServer(manager, bind_host=bind_host, port=port) as server:
        server_url = server.url_for_workers(advertise_host=advertise_host)
        try:
            worker_pool.start(server_url=server_url, token=server.token)

            while not manager.done.wait(timeout=0.1):
                manager.expire_old_leases()
                if error := worker_pool.check():
                    manager.fail(error)
        except BaseException:
            if not manager.done.is_set():
                manager.fail("manager run interrupted before completion")
            raise
        finally:
            worker_pool.stop()

    manager.raise_for_failure()
