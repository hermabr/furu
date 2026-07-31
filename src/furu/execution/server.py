from __future__ import annotations

import http
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hmac import compare_digest
from secrets import token_urlsafe

from pydantic import TypeAdapter
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.sync.server import ServerConnection, serve

from furu.execution.execution_coordinator import ExecutionCoordinator
from furu.worker.protocol import (
    JobResultMessage,
    LeaseJobRequest,
    LeaseJobResponse,
    OkResponse,
    WorkerMessage,
)

_WORKER_MESSAGE = TypeAdapter[WorkerMessage](WorkerMessage)
_LEASE_JOB_RESPONSE = TypeAdapter[LeaseJobResponse](LeaseJobResponse)


@dataclass(frozen=True, slots=True)
class ExecutionCoordinatorServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"ws://{self.bound_host}:{self.bound_port}"


def _handle_worker(
    coordinator: ExecutionCoordinator, connection: ServerConnection
) -> None:
    worker: str | None = None
    try:
        for raw in connection:
            match _WORKER_MESSAGE.validate_json(raw):
                case LeaseJobRequest(resources=resources, worker=worker_name):
                    worker = worker_name
                    lease = coordinator.lease_job(
                        resources=resources, worker=worker_name
                    )
                    connection.send(_LEASE_JOB_RESPONSE.dump_json(lease))
                case JobResultMessage(lease_id=lease_id, result=result):
                    coordinator.job_result(lease_id, result)
                    connection.send(OkResponse().model_dump_json())
    except ConnectionClosed:
        pass
    finally:
        # The connection is the worker's liveness signal: once it drops --
        # cleanly or not -- any leases the worker still holds go back to ready.
        if worker is not None:
            coordinator.worker_lost(worker)


@contextmanager
def execution_coordinator_server(
    coordinator: ExecutionCoordinator, *, bind_host: str, port: int
) -> Iterator[ExecutionCoordinatorServer]:
    auth_token = token_urlsafe(32)

    def require_auth(connection: ServerConnection, request: Request) -> Response | None:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not compare_digest(token, auth_token):
            return connection.respond(
                http.HTTPStatus.UNAUTHORIZED,
                "invalid furu execution coordinator auth token\n",
            )
        return None

    server = serve(
        lambda connection: _handle_worker(coordinator, connection),
        bind_host,
        port,
        process_request=require_auth,
        max_size=None,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name="furu-execution-coordinator-server",
    )
    try:
        thread.start()
        bound_host, bound_port = server.socket.getsockname()[:2]
        yield ExecutionCoordinatorServer(
            bound_host=bound_host,
            bound_port=bound_port,
            auth_token=auth_token,
        )
    finally:
        server.shutdown()
        thread.join(timeout=10)
