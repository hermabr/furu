from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from secrets import token_urlsafe
from typing import cast

import uvicorn

from furu.execution.api import create_manager_api_app
from furu.execution.manager import Manager
from furu.logging import get_logger
from furu.worker.backends import WorkerBackend, WorkerBackendInput, WorkerPool


logger = get_logger()


@dataclass(frozen=True, slots=True)
class ManagerServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"http://{self.bound_host}:{self.bound_port}"


@dataclass(frozen=True, slots=True)
class RunningWorkerPool:
    backend: WorkerBackend
    pool: WorkerPool


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
    worker_backend: WorkerBackendInput,
    port: int,
) -> None:
    worker_backends = _normalize_worker_backends(worker_backend)
    with manager.log_context():
        logger.info(
            "starting furu manager: executor_id=%s executor_dir=%s ready=%d blocked=%d worker_backends=%d",
            manager.executor_id,
            manager.executor_dir,
            len(manager.ready),
            len(manager.blocked),
            len(worker_backends),
        )
        with ExitStack() as stack:
            servers_by_listen_host: dict[str, ManagerServer] = {}
            for listen_host in dict.fromkeys(
                backend.manager_listen_host for backend in worker_backends
            ):
                server = stack.enter_context(
                    manager_server(
                        manager,
                        bind_host=listen_host,
                        port=port,
                    )
                )
                servers_by_listen_host[listen_host] = server
                logger.info(
                    "manager server listening: bind_host=%s server_url=%s",
                    listen_host,
                    server.server_url,
                )

            running_pools: list[RunningWorkerPool] = []
            try:
                for backend in worker_backends:
                    server = servers_by_listen_host[backend.manager_listen_host]
                    worker_pool = backend.start_pool(
                        server_url=server.server_url,
                        auth_token=server.auth_token,
                        executor_dir=manager.executor_dir,
                    )
                    running_pools.append(
                        RunningWorkerPool(backend=backend, pool=worker_pool)
                    )
                    logger.info(
                        "worker pool started: backend=%s health_check_interval=%s",
                        type(backend).__name__,
                        worker_pool.health_check_interval,
                    )
                _wait_until_done(manager, running_pools)
            finally:
                for running_pool in running_pools:
                    running_pool.pool.join(timeout=5)
                    logger.debug(
                        "worker pool joined: backend=%s",
                        type(running_pool.backend).__name__,
                    )
    manager.raise_for_failure()


def _normalize_worker_backends(
    worker_backend: WorkerBackendInput,
) -> tuple[WorkerBackend, ...]:
    if isinstance(worker_backend, Sequence) and not hasattr(
        worker_backend, "start_pool"
    ):
        worker_backends = tuple(worker_backend)
    else:
        worker_backends = (cast(WorkerBackend, worker_backend),)

    if not worker_backends:
        raise ValueError("at least one worker backend is required")
    return worker_backends


def _wait_until_done(
    manager: Manager,
    running_pools: Sequence[RunningWorkerPool],
) -> None:
    if len(running_pools) == 1:
        running_pool = running_pools[0]
        while not manager.done.wait(
            timeout=running_pool.pool.health_check_interval,
        ):
            if not running_pool.pool.is_healthy():
                _fail_unhealthy_pool(manager, running_pool)
                break
        return

    next_health_check_at = [
        time.monotonic() + running_pool.pool.health_check_interval
        for running_pool in running_pools
    ]
    while not manager.done.is_set():
        timeout = max(0.0, min(next_health_check_at) - time.monotonic())
        if manager.done.wait(timeout=timeout):
            break

        now = time.monotonic()
        for idx, running_pool in enumerate(running_pools):
            if now < next_health_check_at[idx]:
                continue
            if not running_pool.pool.is_healthy():
                _fail_unhealthy_pool(manager, running_pool)
                return
            next_health_check_at[idx] = now + running_pool.pool.health_check_interval


def _fail_unhealthy_pool(manager: Manager, running_pool: RunningWorkerPool) -> None:
    manager.fail(
        f"worker backend {type(running_pool.backend).__name__} became unhealthy "
        "before manager run completed"
    )
