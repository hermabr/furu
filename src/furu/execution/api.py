from __future__ import annotations

import secrets
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import TypeAdapter

from furu.execution.manager import Manager
from furu.worker.protocol import (
    LeaseJobResponse,
    JobResultRequest,
    OkResponse,
)

MANAGER_AUTH_HEADER = "x-furu-manager-token"


def generate_manager_auth_token() -> str:
    return secrets.token_urlsafe(32)


class ManagerApiClient:
    def __init__(self, server_url: str, *, auth_token: str) -> None:
        self._server_url = server_url.rstrip("/")
        self._auth_token = auth_token

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
            response = httpx.request(
                method,
                url,
                headers={MANAGER_AUTH_HEADER: self._auth_token},
                json=payload,
                timeout=10.0,
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


def create_manager_api_app(manager: Manager, *, auth_token: str) -> FastAPI:
    app = FastAPI()

    def require_auth(
        supplied_token: str | None = Header(default=None, alias=MANAGER_AUTH_HEADER),
    ) -> None:
        if supplied_token is None or not secrets.compare_digest(
            supplied_token, auth_token
        ):
            raise HTTPException(status_code=401, detail="invalid manager auth token")

    @app.post("/lease_job", response_model=LeaseJobResponse)
    def lease_job(_auth: None = Depends(require_auth)) -> LeaseJobResponse:
        return manager.lease_job()

    @app.post("/job_result/{lease_id}", response_model=OkResponse)
    def job_result(
        lease_id: str,
        request: JobResultRequest,
        _auth: None = Depends(require_auth),
    ) -> OkResponse:
        manager.job_result(lease_id, request)
        return OkResponse()

    return app
