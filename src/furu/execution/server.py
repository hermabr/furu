from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hmac import compare_digest
from http import HTTPStatus
from secrets import token_urlsafe

from pydantic import BaseModel, TypeAdapter, ValidationError
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.sync.server import ServerConnection, serve

from furu.execution.execution_coordinator import ExecutionCoordinator
from furu.worker.protocol import (
    ClientRequest,
    CountSatisfiableJobsRequest,
    CountSatisfiableJobsResponse,
    FailRequest,
    Job,
    JobResponse,
    JobResultMessage,
    LeaseJobRequest,
    OkResponse,
    PoolRequest,
    StopResponse,
    WaitResponse,
    WorkerRequest,
)


@dataclass(frozen=True, slots=True)
class ExecutionCoordinatorServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"ws://{self.bound_host}:{self.bound_port}"


def _send(connection: ServerConnection, message: BaseModel) -> None:
    connection.send(message.model_dump_json())


def _serve_worker(
    connection: ServerConnection,
    coordinator: ExecutionCoordinator,
    initial_request: LeaseJobRequest,
) -> None:
    adapter = TypeAdapter(WorkerRequest)
    resources = initial_request.resources
    worker = initial_request.worker

    def handle(request: WorkerRequest) -> bool:
        match request:
            case LeaseJobRequest(
                resources=request_resources, worker=request_worker
            ):
                if request_resources != resources or request_worker != worker:
                    connection.close(
                        code=1008,
                        reason="worker WebSocket identity cannot change",
                    )
                    return False
                match coordinator.lease_job(resources=resources, worker=worker):
                    case Job() as job:
                        response = JobResponse(job=job)
                    case "wait":
                        response = WaitResponse()
                    case "stop":
                        response = StopResponse()
                _send(connection, response)
            case JobResultMessage(lease_id=lease_id, result=result):
                coordinator.job_result(lease_id, result)
                _send(connection, OkResponse())
        return True

    try:
        handle(initial_request)
        for raw_message in connection:
            request = adapter.validate_json(raw_message)
            if not handle(request):
                return
    finally:
        coordinator.worker_lost(worker)


def _serve_pool(
    connection: ServerConnection,
    coordinator: ExecutionCoordinator,
    initial_request: PoolRequest,
) -> None:
    adapter = TypeAdapter(PoolRequest)

    def handle(request: PoolRequest) -> None:
        match request:
            case CountSatisfiableJobsRequest(
                resources=resources, max_workers=max_workers
            ):
                _send(
                    connection,
                    CountSatisfiableJobsResponse(
                        count=coordinator.count_satisfiable_jobs(
                            resources=resources, max_workers=max_workers
                        )
                    ),
                )
            case FailRequest(message=message):
                coordinator.fail(message)
                _send(connection, OkResponse())

    handle(initial_request)
    for raw_message in connection:
        handle(adapter.validate_json(raw_message))


def _handle_connection(
    connection: ServerConnection, coordinator: ExecutionCoordinator
) -> None:
    try:
        request = TypeAdapter(ClientRequest).validate_json(connection.recv())
        match request:
            case LeaseJobRequest():
                _serve_worker(connection, coordinator, request)
            case CountSatisfiableJobsRequest() | FailRequest():
                _serve_pool(connection, coordinator, request)
            case JobResultMessage():
                connection.close(code=1008, reason="job result before worker lease")
    except ValidationError:
        connection.close(code=1003, reason="invalid typed message")
    except ConnectionClosed:
        pass


@contextmanager
def execution_coordinator_server(
    coordinator: ExecutionCoordinator, *, bind_host: str, port: int
) -> Iterator[ExecutionCoordinatorServer]:
    auth_token = token_urlsafe(32)
    websocket_server = None
    thread: threading.Thread | None = None

    def require_auth(
        connection: ServerConnection, request: Request
    ) -> Response | None:
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not compare_digest(token, auth_token):
            return connection.respond(
                HTTPStatus.UNAUTHORIZED,
                "invalid furu execution coordinator auth token\n",
            )
        return None

    try:
        websocket_server = serve(
            lambda connection: _handle_connection(connection, coordinator),
            host=bind_host,
            port=port,
            process_request=require_auth,
            ping_interval=10,
            ping_timeout=10,
        )
        bound_host, bound_port = websocket_server.socket.getsockname()[:2]
        thread = threading.Thread(
            target=websocket_server.serve_forever,
            name="furu-execution-coordinator-server",
        )
        thread.start()

        yield ExecutionCoordinatorServer(
            bound_host=bound_host,
            bound_port=bound_port,
            auth_token=auth_token,
        )
    finally:
        if websocket_server is not None:
            websocket_server.shutdown()
        if thread is not None:
            thread.join(timeout=10)
