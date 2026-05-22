from __future__ import annotations

from hmac import compare_digest
from typing import Any

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from pydantic import TypeAdapter

from furu.execution.manager import Manager
from furu.resources import ResourceRequest
from furu.worker.protocol import (
    CountSatisfiableJobsRequest,
    FailRequest,
    LeaseJobRequest,
    LeaseJobResponse,
    JobResultRequest,
    OkResponse,
)


class _ManagerApiClientBase:
    def __init__(self, server_url: str, *, auth_token: str) -> None:
        self._server_url = server_url.rstrip("/")
        self._auth_token = auth_token

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


class WorkerApiClient(_ManagerApiClientBase):
    def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
        response = self._request_json(
            "/worker/lease_job",
            method="POST",
            payload=LeaseJobRequest(resources=resources).model_dump(mode="json"),
        )
        return TypeAdapter(LeaseJobResponse).validate_python(response)

    def job_result(self, lease_id: str, request: JobResultRequest) -> None:
        response = self._request_json(
            f"/worker/job_result/{lease_id}",
            method="POST",
            payload=request.model_dump(mode="json"),
        )
        OkResponse.model_validate(response)


class PoolApiClient(_ManagerApiClientBase):
    def count_satisfiable_jobs(
        self, *, resources: ResourceRequest, max_workers: int
    ) -> int:
        response = self._request_json(
            "/pool/count_satisfiable_jobs",
            method="POST",
            payload=CountSatisfiableJobsRequest(
                resources=resources, max_workers=max_workers
            ).model_dump(mode="json"),
        )
        return int(response)

    def fail(self, *, message: str) -> None:
        response = self._request_json(
            "/pool/fail",
            method="POST",
            payload=FailRequest(message=message).model_dump(mode="json"),
        )
        OkResponse.model_validate(response)


def create_manager_api_app(manager: Manager, *, auth_token: str) -> FastAPI:
    def require_auth(authorization: str = Header(default="")) -> None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not compare_digest(token, auth_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid furu manager auth token",
            )

    app = FastAPI()
    auth_dependency = Depends(require_auth)

    worker_router = APIRouter(prefix="/worker", dependencies=[auth_dependency])

    @worker_router.post("/lease_job", response_model=LeaseJobResponse)
    def lease_job(request: LeaseJobRequest) -> LeaseJobResponse:
        return manager.lease_job(resources=request.resources)

    @worker_router.post("/job_result/{lease_id}", response_model=OkResponse)
    def job_result(lease_id: str, request: JobResultRequest) -> OkResponse:
        manager.job_result(lease_id, request)
        return OkResponse()

    pool_router = APIRouter(prefix="/pool", dependencies=[auth_dependency])

    @pool_router.post("/count_satisfiable_jobs")
    def count_satisfiable_jobs(request: CountSatisfiableJobsRequest) -> int:
        return manager.count_satisfiable_jobs(
            resources=request.resources, max_workers=request.max_workers
        )

    @pool_router.post("/fail", response_model=OkResponse)
    def fail(request: FailRequest) -> OkResponse:
        manager.fail(request.message)
        return OkResponse()

    app.include_router(worker_router)
    app.include_router(pool_router)
    return app
