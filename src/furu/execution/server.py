from __future__ import annotations

import socket
import threading
import time
from collections.abc import Sequence

import uvicorn

from furu.execution.api import create_manager_api_app
from furu.execution.manager import Manager


def _run_until_done(
    manager: Manager,
    *,
    n_workers: int,
    host: str = "127.0.0.1",
    port: int = 0,
) -> None:
    from furu.worker.loop import worker_loop

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
    workers: Sequence[threading.Thread] = [
        threading.Thread(
            target=worker_loop,
            kwargs={"server_url": server_url},
            name=f"furu-worker-{idx}",
        )
        for idx in range(n_workers)
    ]

    try:
        server_thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not server_thread.is_alive():
                raise RuntimeError("manager server exited before it was ready")
            if time.monotonic() > deadline:
                raise TimeoutError("manager server did not start within 10 seconds")
            time.sleep(0.01)

        for worker in workers:
            worker.start()

        while not manager.done.wait(timeout=0.1):
            if any(not worker.is_alive() for worker in workers):
                manager.fail("a worker exited before manager run completed")
                break

        for worker in workers:
            worker.join(timeout=5)
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)

    manager.raise_for_failure()
