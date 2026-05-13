from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import TypeAdapter

from furu.execution.manager import Manager
from furu.worker.protocol import (
    LeaseJobResponse,
    JobResultRequest,
    OkResponse,
)


class ManagerApiClient:
    def __init__(self, server_url: str) -> None:
        self._server_url = server_url.rstrip("/")

    def lease_job(self) -> LeaseJobResponse:
        response = self._request_json("/lease_job", method="POST")
        return TypeAdapter(LeaseJobResponse).validate_python(response)

    def job_result(self, lease_id: str, request: JobResultRequest) -> None:
        response = self._request_json(
            f"/job_result/{lease_id}",
            method="POST",
            payload=request.model_dump(mode="json"),
        )
        OkResponse.model_validate(response)

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: object | None = None,
    ) -> Any:
        url = f"{self._server_url}{path}"
        try:
            response = httpx.request(method, url, json=payload, timeout=10.0)
            response.raise_for_status()
            if not response.content:
                raise RuntimeError(f"{method} {url} returned an empty response")
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"{method} {url} failed with HTTP "
                f"{exc.response.status_code}: {exc.response.text}"
            ) from exc


def create_manager_api_app(manager: Manager) -> FastAPI:
    app = FastAPI()

    @app.post("/lease_job", response_model=LeaseJobResponse)
    def lease_job() -> LeaseJobResponse:
        return manager.lease_job()

    @app.post("/job_result/{lease_id}", response_model=OkResponse)
    def job_result(lease_id: str, request: JobResultRequest) -> OkResponse:
        manager.job_result(lease_id, request)
        return OkResponse()

    return app
