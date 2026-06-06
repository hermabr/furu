from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from secrets import token_urlsafe

import uvicorn

from furu.execution.api import create_execution_coordinator_api_app
from furu.execution.coordinator import ExecutionCoordinator


@dataclass(frozen=True, slots=True)
class ExecutionCoordinatorServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"http://{self.bound_host}:{self.bound_port}"


@contextmanager
def execution_coordinator_server(
    coordinator: ExecutionCoordinator, *, bind_host: str, port: int
) -> Iterator[ExecutionCoordinatorServer]:
    auth_token = token_urlsafe(32)
    app = create_execution_coordinator_api_app(coordinator, auth_token=auth_token)
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
            name="furu-execution-coordinator-server",
        )
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError(
                    "execution coordinator server exited before it was ready"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "execution coordinator server did not start within 10 seconds"
                )
            time.sleep(0.01)

        yield ExecutionCoordinatorServer(
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
