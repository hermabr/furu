from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter
from websockets.exceptions import WebSocketException
from websockets.sync.client import ClientConnection, connect

from furu.resources import ResourceRequest
from furu.worker.protocol import (
    JobResult,
    JobResultMessage,
    LeaseJobRequest,
    LeaseJobResponse,
    OkResponse,
)

_LEASE_JOB_RESPONSE = TypeAdapter[LeaseJobResponse](LeaseJobResponse)


@dataclass(frozen=True, slots=True)
class WorkerClient:
    """Lockstep request/response client over one WebSocket connection.

    The connection doubles as the worker's liveness signal: if it drops for
    any reason, the execution coordinator releases every lease this worker
    holds. Keepalive pings detect dead peers on both sides.
    """

    _connection: ClientConnection
    request_timeout_s: float = 10.0

    @classmethod
    @contextmanager
    def connect(cls, *, server_url: str, auth_token: str) -> Iterator[WorkerClient]:
        try:
            connection = connect(
                server_url,
                additional_headers={"Authorization": f"Bearer {auth_token}"},
                max_size=None,
            )
        except (OSError, WebSocketException) as exc:
            raise RuntimeError(
                f"connecting to execution coordinator at {server_url} failed: {exc}"
            ) from exc
        with connection:
            yield cls(_connection=connection)

    def _request(self, message: BaseModel) -> str | bytes:
        try:
            self._connection.send(message.model_dump_json())
            return self._connection.recv(timeout=self.request_timeout_s)
        except (OSError, TimeoutError, WebSocketException) as exc:
            raise RuntimeError(
                f"execution coordinator connection failed: {exc!r}"
            ) from exc

    def lease_job(self, *, resources: ResourceRequest, worker: str) -> LeaseJobResponse:
        # Validate the raw JSON body: Job's strict models accept datetimes and
        # tuples only in JSON mode, not from an already-parsed dict.
        response = self._request(LeaseJobRequest(resources=resources, worker=worker))
        return _LEASE_JOB_RESPONSE.validate_json(response)

    def job_result(self, lease_id: str, result: JobResult) -> None:
        response = self._request(JobResultMessage(lease_id=lease_id, result=result))
        OkResponse.model_validate_json(response)
