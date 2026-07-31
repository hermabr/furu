from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, TypeAdapter
from websockets.exceptions import WebSocketException
from websockets.sync.client import ClientConnection, connect

from furu.resources import ResourceRequest
from furu.worker.protocol import (
    CountSatisfiableJobsRequest,
    CountSatisfiableJobsResponse,
    FailRequest,
    JobResponse,
    JobResultMessage,
    JobResultRequest,
    LeaseJobRequest,
    LeaseJobResponse,
    LeaseJobWireResponse,
    OkResponse,
    StopResponse,
    WaitResponse,
)


@dataclass(slots=True, kw_only=True)
class _ExecutionCoordinatorApiClientBase:
    server_url: str
    auth_token: str
    request_timeout_s: float = 10.0
    _connection: ClientConnection | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def _connect(self) -> ClientConnection:
        if self._closed:
            raise RuntimeError("execution coordinator WebSocket client is closed")
        if self._connection is not None:
            return self._connection
        try:
            connection = connect(
                self.server_url,
                additional_headers={
                    "Authorization": f"Bearer {self.auth_token}",
                },
                open_timeout=self.request_timeout_s,
                close_timeout=self.request_timeout_s,
            )
        except (OSError, WebSocketException, ValueError) as exc:
            raise RuntimeError(
                f"WebSocket connection to {self.server_url} failed: {exc}"
            ) from exc
        self._connection = connection
        return connection

    def _request[ResponseT](
        self, request: BaseModel, response_type: Any
    ) -> ResponseT:
        connection = self._connect()
        try:
            connection.send(request.model_dump_json())
            return TypeAdapter(response_type).validate_json(
                connection.recv(timeout=self.request_timeout_s)
            )
        except (OSError, WebSocketException, ValueError) as exc:
            self.close()
            raise RuntimeError(
                f"WebSocket request to {self.server_url} failed: {exc}"
            ) from exc

    def close(self) -> None:
        self._closed = True
        if self._connection is not None:
            self._connection.close()
            self._connection = None


@dataclass(slots=True, kw_only=True)
class WorkerApiClient(_ExecutionCoordinatorApiClientBase):
    def lease_job(self, *, resources: ResourceRequest, worker: str) -> LeaseJobResponse:
        response = self._request(
            LeaseJobRequest(resources=resources, worker=worker), LeaseJobWireResponse
        )
        match response:
            case JobResponse(job=job):
                return job
            case WaitResponse():
                return "wait"
            case StopResponse():
                return "stop"
        raise AssertionError(f"unexpected lease response: {response!r}")

    def job_result(self, lease_id: str, request: JobResultRequest) -> None:
        response = self._request(
            JobResultMessage(lease_id=lease_id, result=request), OkResponse
        )
        assert isinstance(response, OkResponse)


@dataclass(slots=True, kw_only=True)
class PoolApiClient(_ExecutionCoordinatorApiClientBase):
    def count_satisfiable_jobs(
        self, *, resources: ResourceRequest, max_workers: int
    ) -> int:
        response = self._request(
            CountSatisfiableJobsRequest(
                resources=resources, max_workers=max_workers
            ),
            CountSatisfiableJobsResponse,
        )
        assert isinstance(response, CountSatisfiableJobsResponse)
        return response.count

    def fail(self, *, message: str) -> None:
        response = self._request(FailRequest(message=message), OkResponse)
        assert isinstance(response, OkResponse)
