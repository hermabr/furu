from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hmac import compare_digest
from http import HTTPStatus
from secrets import token_urlsafe

from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.sync.server import ServerConnection, serve

from furu.execution.execution_coordinator import ExecutionCoordinator
from furu.logging import get_logger, log_detail
from furu.worker.protocol import (
    PROTOCOL_VERSION,
    AssignMessage,
    HelloMessage,
    Job,
    ResultMessage,
    StopMessage,
    WelcomeMessage,
    worker_message_adapter,
)

logger = get_logger()

_HELLO_TIMEOUT_S = 10.0
# Fallback re-check while a worker waits for work, in case a wake
# notification races with entering the wait.  Dispatch is normally instant.
_WAIT_RECHECK_S = 1.0


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
    stopping: threading.Event,
) -> None:
    with coordinator.log_context():
        match worker_message_adapter.validate_json(
            connection.recv(timeout=_HELLO_TIMEOUT_S)
        ):
            case HelloMessage() as hello:
                pass
            case unexpected:
                raise RuntimeError(f"expected hello message, got {unexpected.type!r}")
        if hello.version != PROTOCOL_VERSION:
            connection.send(
                StopMessage(
                    reason=(
                        f"protocol version mismatch: worker speaks {hello.version}, "
                        f"coordinator speaks {PROTOCOL_VERSION}"
                    )
                ).model_dump_json()
            )
            return
        worker = hello.worker
        logger.info(
            "worker connected · %s",
            worker,
            extra=log_detail(worker=worker, backend=hello.backend),
        )
        try:
            connection.send(
                WelcomeMessage(executor_id=coordinator.executor_id).model_dump_json()
            )
            while not stopping.is_set():
                match coordinator.lease_job(resources=hello.resources, worker=worker):
                    case "stop":
                        connection.send(
                            StopMessage(reason="run finished").model_dump_json()
                        )
                        return
                    case "wait":
                        coordinator.wait_for_state_change(timeout=_WAIT_RECHECK_S)
                    case Job() as job:
                        connection.send(AssignMessage(job=job).model_dump_json())
                        match worker_message_adapter.validate_json(connection.recv()):
                            case ResultMessage() as reply:
                                pass
                            case unexpected:
                                raise RuntimeError(
                                    f"expected result message, got {unexpected.type!r}"
                                )
                        for member in job.members:
                            coordinator.job_result(member.lease_id, reply.result)
            connection.send(
                StopMessage(reason="coordinator shutting down").model_dump_json()
            )
        except ConnectionClosed:
            logger.warning(
                "worker disconnected · %s",
                worker,
                extra=log_detail(worker=worker),
            )
        finally:
            coordinator.worker_lost(worker)


@contextmanager
def execution_coordinator_server(
    coordinator: ExecutionCoordinator, *, bind_host: str, port: int
) -> Iterator[ExecutionCoordinatorServer]:
    auth_token = token_urlsafe(32)
    stopping = threading.Event()
    connections: set[ServerConnection] = set()
    connections_lock = threading.Lock()

    def process_request(
        connection: ServerConnection, request: Request
    ) -> Response | None:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not compare_digest(token, auth_token):
            return connection.respond(
                HTTPStatus.UNAUTHORIZED,
                "invalid furu execution coordinator auth token\n",
            )
        return None

    def handler(connection: ServerConnection) -> None:
        with connections_lock:
            connections.add(connection)
        try:
            _serve_worker(coordinator, connection, stopping)
        finally:
            with connections_lock:
                connections.discard(connection)

    server = serve(
        handler,
        bind_host,
        port,
        process_request=process_request,
        max_size=None,
    )
    bound_host, bound_port = server.socket.getsockname()[:2]
    thread = threading.Thread(
        target=server.serve_forever,
        name="furu-execution-coordinator-server",
    )
    thread.start()
    try:
        yield ExecutionCoordinatorServer(
            bound_host=bound_host,
            bound_port=bound_port,
            auth_token=auth_token,
        )
    finally:
        stopping.set()
        # Wake handlers idling in wait_for_state_change so they say stop and
        # exit; shutdown() only stops accepting, so also close what is open to
        # unblock handlers waiting on a result from a busy or dead worker.
        coordinator.notify_state_changed()
        server.shutdown()
        thread.join(timeout=10)
        with connections_lock:
            open_connections = tuple(connections)
        for connection in open_connections:
            connection.close()
