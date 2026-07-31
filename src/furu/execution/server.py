from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from secrets import token_urlsafe

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, basic_auth, serve

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
        match worker_message_adapter.validate_json(connection.recv(timeout=10.0)):
            case HelloMessage() as hello:
                pass
            case unexpected:
                raise RuntimeError(f"expected hello message, got {unexpected.kind!r}")
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
            while True:
                match coordinator.lease_job(resources=hello.resources, worker=worker):
                    case "stop":
                        connection.send(
                            StopMessage(reason="run finished").model_dump_json()
                        )
                        return
                    case "wait":
                        with coordinator.wake:
                            coordinator.wake.wait(timeout=1.0)
                    case Job() as job:
                        connection.send(AssignMessage(job=job).model_dump_json())
                        match worker_message_adapter.validate_json(connection.recv()):
                            case ResultMessage() as reply:
                                pass
                            case unexpected:
                                raise RuntimeError(
                                    f"expected result message, got {unexpected.kind!r}"
                                )
                        for member in job.members:
                            coordinator.job_result(member.lease_id, reply.result)
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
    connections: set[ServerConnection] = set()
    connections_changed = threading.Condition()

    def handler(connection: ServerConnection) -> None:
        with connections_changed:
            connections.add(connection)
        try:
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
        yield ExecutionCoordinatorServer(
            bound_host=bound_host,
            bound_port=bound_port,
            auth_token=auth_token,
        )
    finally:
        server.shutdown()
        thread.join(timeout=10)
        with connections_changed:
            open_connections = tuple(connections)
        for connection in open_connections:
            connection.close()
        with connections_changed:
            connections_changed.wait_for(lambda: not connections, timeout=10)
