from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from secrets import token_urlsafe

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, basic_auth, serve

from furu.execution.execution_coordinator import ExecutionCoordinator
from furu.logging import get_logger, log_detail
from furu.worker.protocol import (
    CancelMessage,
    HelloMessage,
    PoolHandoff,
    TakeoverRequest,
    TakeoverResponse,
    first_message_adapter,
    job_result_adapter,
)

_TAKEOVER_REPLY_TIMEOUT_S = 120.0

logger = get_logger()


@dataclass(frozen=True, slots=True)
class ExecutionCoordinatorServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"ws://{self.bound_host}:{self.bound_port}"


def _serve(coordinator: ExecutionCoordinator, connection: ServerConnection) -> None:
    with coordinator.log_context():
        match first_message_adapter.validate_json(connection.recv(timeout=10.0)):
            case HelloMessage() as hello:
                _serve_worker(coordinator, connection, hello)
            case TakeoverRequest() as request:
                _serve_takeover(coordinator, connection, request)


def _serve_takeover(
    coordinator: ExecutionCoordinator,
    connection: ServerConnection,
    request: TakeoverRequest,
) -> None:
    handoffs = {
        key: coordinator.pools[key].handoff()
        for key in request.pool_keys
        if key in coordinator.pools
    }
    logger.info(
        "handed off %d of %d pools to exec=%s",
        len(handoffs),
        len(coordinator.pools),
        request.executor_id[:5],
    )
    connection.send(TakeoverResponse(handoffs=handoffs).model_dump_json())
    # The new coordinator closes the connection once the inherited workers are
    # pointed at it. Whether it got that far or died trying, our part is over:
    # handed-off pools have nothing left to cancel and the rest stop as usual.
    with suppress(ConnectionClosed):
        connection.recv()
    coordinator.fail(f"execution taken over by exec={request.executor_id[:5]}")


@contextmanager
def request_takeover(
    *,
    executor_id: str,
    source_id: str,
    url: str,
    pool_keys: Sequence[str],
) -> Iterator[dict[str, PoolHandoff]]:
    """Inherit ``source_id``'s matching pools; closing the connection commits."""
    try:
        connection = connect(url, max_size=None)
    except OSError as exc:
        raise RuntimeError(
            f"cannot reach exec={source_id[:5]}; is that coordinator still running?"
        ) from exc
    with connection:
        connection.send(
            TakeoverRequest(
                executor_id=executor_id, pool_keys=list(pool_keys)
            ).model_dump_json()
        )
        response = TakeoverResponse.model_validate_json(
            connection.recv(timeout=_TAKEOVER_REPLY_TIMEOUT_S)
        )
        yield response.handoffs


def _serve_worker(
    coordinator: ExecutionCoordinator,
    connection: ServerConnection,
    hello: HelloMessage,
) -> None:
    with coordinator.log_context():
        worker = hello.worker
        logger.info(
            "worker connected · %s%s",
            worker,
            f" · running {hello.running[0].log_label}" if hello.running else "",
            extra=log_detail(worker=worker, backend=hello.backend),
        )
        try:
            if hello.running:
                if not coordinator.adopt(hello.running, worker=worker):
                    connection.send(CancelMessage().model_dump_json())
                result = job_result_adapter.validate_json(connection.recv())
                for artifact in hello.running:
                    coordinator.job_result(artifact.object_id, result)
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
            _serve(coordinator, connection)
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
        coordinator.fail("execution coordinator server closed before the run finished")
        server.shutdown()
        thread.join(timeout=10)
        with connections_changed:
            open_connections = tuple(connections)
        for connection in open_connections:
            connection.close()
        with connections_changed:
            connections_changed.wait_for(lambda: not connections, timeout=10)
