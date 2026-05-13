from __future__ import annotations

import socket
import threading
import time

import uvicorn

from furu.execution.api import create_manager_api_app
from furu.execution.manager import Manager
from furu.worker.backends import WorkerBackend


def _run_until_done(
    manager: Manager,
    *,
    worker_backend: WorkerBackend,
    host: str,
    port: int,
) -> None:
    app = create_manager_api_app(manager)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen()
    sock.set_inheritable(True)
    bound_host, bound_port = sock.getsockname()[:2]
    server_url = f"http://{bound_host}:{bound_port}"

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
    worker_pool = worker_backend.create_pool(server_url=server_url)

    try:
        server_thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not server_thread.is_alive():
                raise RuntimeError("manager server exited before it was ready")
            if time.monotonic() > deadline:
                raise TimeoutError("manager server did not start within 10 seconds")
            time.sleep(0.01)

        worker_pool.start()

        while not manager.done.wait(timeout=0.1):
            if not worker_pool.is_healthy():
                manager.fail(
                    "worker backend became unhealthy before manager run completed"
                )
                break

        worker_pool.join(timeout=5)
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)

    manager.raise_for_failure()
