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
) -> None:
    (bind_host,) = {backend.manager_listen_host for backend in worker_backends}

    with manager.log_context():
        logger.info(
            "starting furu manager: executor_id=%s executor_dir=%s ready=%d blocked=%d",
            manager.executor_id,
            manager.executor_dir,
            len(manager.ready),
            len(manager.blocked),
        )
        with manager_server(manager, bind_host=bind_host, port=port) as server:
            logger.info(
                "manager server listening: server_url=%s",
                server.server_url,
            )
            worker_pools: list[tuple[WorkerBackend, WorkerPool]] = []
            for backend in worker_backends:
                pool = backend.start_pool(
                    server_url=server.server_url,
                    auth_token=server.auth_token,
                    executor_dir=manager.executor_dir,
                )
                worker_pools.append((backend, pool))
                logger.info(
                    "worker pool started: backend=%s health_check_interval=%s",
                    type(backend).__name__,
                    pool.health_check_interval,
                )
            next_check_at = [
                time.monotonic() + pool.health_check_interval
                for _, pool in worker_pools
            ]
            while not manager.done.wait(
                timeout=max(0.0, min(next_check_at) - time.monotonic())
            ):
                now = time.monotonic()
                unhealthy: list[str] = []
                for idx, (backend, pool) in enumerate(worker_pools):
                    if now < next_check_at[idx]:
                        continue
                    pool.scale()
                    if not pool.is_healthy():
                        unhealthy.append(type(backend).__name__)
                    next_check_at[idx] = now + pool.health_check_interval
                if unhealthy:
                    manager.fail(
                        "worker backend(s) became unhealthy before manager "
                        f"run completed: {', '.join(unhealthy)}"
                    )
                    break
            for backend, pool in worker_pools:
                pool.join(timeout=5)
                logger.debug("worker pool joined: backend=%s", type(backend).__name__)
    manager.raise_for_failure()
