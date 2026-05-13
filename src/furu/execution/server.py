from __future__ import annotations

import socket
import threading
import time
from collections.abc import Sequence

import uvicorn
from fastapi import FastAPI

from furu.execution.manager import Manager
from furu.worker.protocol import BlockedRequest, FinishRequest, GetJobResponse


def make_app(manager: Manager) -> FastAPI:
    app = FastAPI()

    @app.get("/get_job", response_model=GetJobResponse)
    def get_job() -> GetJobResponse:
        return manager.get_job()

    @app.post("/finish/{lease_id}")
    def finish(lease_id: str, request: FinishRequest) -> dict[str, bool]:
        manager.finish(lease_id, request)
        return {"ok": True}

    @app.post("/blocked/{lease_id}")
    def blocked(lease_id: str, request: BlockedRequest) -> dict[str, bool]:
        manager.report_blocked(lease_id, request.dependencies)
        return {"ok": True}

    return app


def run_until_done(
    manager: Manager,
    *,
    n_workers: int,
    host: str = "127.0.0.1",
    port: int = 0,
) -> None:
    with manager.lock:
        manager._maybe_finish_locked()
    if manager.done.is_set():
        manager.raise_for_failure()
        return

    from furu.worker.loop import worker_loop

    app = make_app(manager)
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
