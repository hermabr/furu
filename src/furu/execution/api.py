from __future__ import annotations

from hmac import compare_digest
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
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
        try:
            response = httpx.request(
                method,
                url,
                headers={"Authorization": f"Bearer {self._auth_token}"},
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


def _authorize_manager_request(
    expected_token: str,
):
    def dependency(authorization: str | None = Header(default=None)) -> None:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not compare_digest(token, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid furu manager auth token",
            )

    return dependency


def create_manager_api_app(manager: Manager, *, auth_token: str) -> FastAPI:
    if not auth_token:
        raise ValueError("manager auth_token must not be empty")

    app = FastAPI()
    auth_dependency = Depends(_authorize_manager_request(auth_token))

    @app.post(
        "/lease_job",
        response_model=LeaseJobResponse,
        dependencies=[auth_dependency],
    )
    def lease_job() -> LeaseJobResponse:
        return manager.lease_job()

    @app.post(
        "/job_result/{lease_id}",
        response_model=OkResponse,
        dependencies=[auth_dependency],
    )
    def job_result(lease_id: str, request: JobResultRequest) -> OkResponse:
        manager.job_result(lease_id, request)
        return OkResponse()

    return app
