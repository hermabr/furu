from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import TypeAdapter

from furu.execution.manager import Manager
from furu.worker.protocol import (
    BlockedRequest,
    FinishRequest,
    GetJobResponse,
    OkResponse,
)


class ManagerApiClient:
    def __init__(self, server_url: str) -> None:
        self._server_url = server_url.rstrip("/")

    def get_job(self) -> GetJobResponse:
        response = self._request_json("/get_job")
        return TypeAdapter(GetJobResponse).validate_python(response)

    def finish(self, lease_id: str, request: FinishRequest) -> None:
        response = self._request_json(
            f"/finish/{lease_id}",
            method="POST",
            payload=request.model_dump(mode="json"),
        )
        OkResponse.model_validate(response)

    def report_blocked(
        self,
        lease_id: str,
        request: BlockedRequest,
    ) -> None:
        response = self._request_json(
            f"/report_blocked/{lease_id}",
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

    @app.get("/get_job", response_model=GetJobResponse)
    def get_job() -> GetJobResponse:
        return manager.get_job()

    @app.post("/finish/{lease_id}", response_model=OkResponse)
    def finish(lease_id: str, request: FinishRequest) -> OkResponse:
        manager.finish(lease_id, request)
        return OkResponse()

    @app.post("/report_blocked/{lease_id}", response_model=OkResponse)
    def report_blocked(lease_id: str, request: BlockedRequest) -> OkResponse:
        manager.report_blocked(lease_id, request.dependencies)
        return OkResponse()

    return app
