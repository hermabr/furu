from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from secrets import token_urlsafe

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, basic_auth, serve

from furu.execution.execution_coordinator import ExecutionCoordinator
from furu.execution.takeover import (
    TAKEOVER_PATH,
    TakeoverRequest,
    register_live_run,
)
from furu.logging import get_logger, log_detail
from furu.worker.protocol import HelloMessage, job_result_adapter

logger = get_logger()


@dataclass(frozen=True, slots=True)
class ExecutionCoordinatorServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"ws://{self.bound_host}:{self.bound_port}"


def _serve_worker(
    coordinator: ExecutionCoordinator,
    connection: ServerConnection,
) -> None:
    with coordinator.log_context():
        hello = HelloMessage.model_validate_json(connection.recv(timeout=10.0))
        worker = hello.worker
        logger.info(
            "worker connected · %s",
            worker,
            extra=log_detail(worker=worker, backend=hello.backend),
        )
        try:
            while True:
                job = coordinator.lease_job(resources=hello.resources, worker=worker)
                if job is None:
                    return
                connection.send(job.model_dump_json())
                result = job_result_adapter.validate_json(connection.recv())
                for artifact in job.artifacts:
                    coordinator.job_result(artifact.object_id, result)
        except ConnectionClosed:
            logger.warning(
                "worker disconnected · %s",
                worker,
                extra=log_detail(worker=worker),
            )
        finally:
            coordinator.worker_lost(worker)


def _serve_takeover(
    coordinator: ExecutionCoordinator,
    connection: ServerConnection,
    busy: threading.Lock,
) -> None:
    """Single request/response exchange with a successor run.

    The request carries everything the takeover needs, so a dropped or
    malformed connection surrenders nothing; once the request validates, this
    run redirects the matched pools itself and only then answers and stops.
    """
    with coordinator.log_context():
        if not busy.acquire(blocking=False):
            logger.warning("rejecting takeover connection: one already in progress")
            connection.close(1013, "takeover already in progress")
            return
        try:
            request = TakeoverRequest.model_validate_json(
                connection.recv(timeout=10.0)
            )
            logger.info(
                "takeover requested by successor %s", request.successor_executor_id
            )
            response = coordinator.handle_takeover(request)
            connection.send(response.model_dump_json())
            if response.adopted:
                coordinator.replaced(request.successor_executor_id)
        finally:
            busy.release()


@contextmanager
def execution_coordinator_server(
    coordinator: ExecutionCoordinator, *, bind_host: str, port: int
) -> Iterator[ExecutionCoordinatorServer]:
    auth_token = token_urlsafe(32)
    connections: set[ServerConnection] = set()
    connections_changed = threading.Condition()
    takeover_busy = threading.Lock()

    def handler(connection: ServerConnection) -> None:
        with connections_changed:
            connections.add(connection)
        try:
            if (
                connection.request is not None
                and connection.request.path == TAKEOVER_PATH
            ):
                _serve_takeover(coordinator, connection, takeover_busy)
            else:
                _serve_worker(coordinator, connection)
        finally:
            with connections_changed:
                connections.discard(connection)
                connections_changed.notify_all()

    server = serve(
        handler,
        bind_host,
        port,
        process_request=basic_auth(credentials=("furu", auth_token)),
        max_size=None,
    )
    bound_host, bound_port = server.socket.getsockname()[:2]
    thread = threading.Thread(
        target=server.serve_forever,
        name="furu-execution-coordinator-server",
    )
    thread.start()
    try:
        with register_live_run(
            executor_id=coordinator.executor_id,
            executor_dir=coordinator.executor_dir,
            bound_port=bound_port,
            auth_token=auth_token,
        ):
            yield ExecutionCoordinatorServer(
                bound_host=bound_host,
                bound_port=bound_port,
                auth_token=auth_token,
            )
    finally:
        coordinator.fail("execution coordinator server closed before the run finished")
        server.shutdown()
        thread.join(timeout=10)
        with connections_changed:
            open_connections = tuple(connections)
        for connection in open_connections:
            connection.close()
        with connections_changed:
            connections_changed.wait_for(lambda: not connections, timeout=10)
