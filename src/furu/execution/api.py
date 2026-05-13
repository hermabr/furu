from __future__ import annotations

import hmac
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
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        try:
            response = httpx.request(
                method, url, json=payload, headers=headers, timeout=10.0
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

    def require_auth(authorization: str = Header(default="")) -> None:
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            credentials, auth_token
        ):
            raise HTTPException(status_code=401, detail="invalid auth token")

    @app.post(
        "/lease_job",
        response_model=LeaseJobResponse,
        dependencies=[Depends(require_auth)],
    )
    def lease_job() -> LeaseJobResponse:
        return manager.lease_job()

    @app.post(
        "/job_result/{lease_id}",
        response_model=OkResponse,
        dependencies=[Depends(require_auth)],
    )
    def job_result(lease_id: str, request: JobResultRequest) -> OkResponse:
        manager.job_result(lease_id, request)
        return OkResponse()

    return app
