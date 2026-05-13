from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import TypeAdapter

from furu.execution.manager import Manager
from furu.worker.protocol import (
    LeaseJobResponse,
    JobResultRequest,
    OkResponse,
)


MANAGER_TOKEN_HEADER = "x-furu-manager-token"


class ManagerApiClient:
    def __init__(self, server_url: str, *, token: str | None = None) -> None:
        self._server_url = server_url.rstrip("/")
        self._token = token

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
            if self._token is None:
                response = httpx.request(method, url, json=payload, timeout=10.0)
            else:
                response = httpx.request(
                    method,
                    url,
                    json=payload,
                    timeout=10.0,
                    headers={MANAGER_TOKEN_HEADER: self._token},
                )
            response.raise_for_status()
            if not response.content:
                raise RuntimeError(f"{method} {url} returned an empty response")
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"{method} {url} failed with HTTP "
                f"{exc.response.status_code}: {exc.response.text}"
            ) from exc


def create_manager_api_app(manager: Manager, *, token: str | None = None) -> FastAPI:
    app = FastAPI()

    def verify_token(request: Request) -> None:
        if token is None:
            return
        if request.headers.get(MANAGER_TOKEN_HEADER) == token:
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid manager token",
        )

    @app.post("/lease_job", response_model=LeaseJobResponse)
    def lease_job(request: Request) -> LeaseJobResponse:
        verify_token(request)
        return manager.lease_job()

    @app.post("/job_result/{lease_id}", response_model=OkResponse)
    def job_result(
        lease_id: str,
        request: Request,
        result: JobResultRequest,
    ) -> OkResponse:
        verify_token(request)
        manager.job_result(lease_id, result)
        return OkResponse()

    return app
