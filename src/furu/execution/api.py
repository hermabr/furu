from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hmac import compare_digest
from typing import Any

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from pydantic import TypeAdapter

from furu.execution.execution_coordinator import ExecutionCoordinator
from furu.logging import _CURRENT_COMPONENT
from furu.resources import ResourceRequest
from furu.worker.protocol import (
    CountSatisfiableJobsRequest,
    FailRequest,
    JobResultRequest,
    LeaseJobRequest,
    LeaseJobResponse,
    OkResponse,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ExecutionCoordinatorApiClientBase:
    server_url: str
    auth_token: str
    request_timeout_s: float = 10.0

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        payload: object | None = None,
    ) -> Any:
        url = f"{self.server_url.rstrip('/')}{path}"
        try:
            response = httpx.request(
                method,
                url,
                headers={"Authorization": f"Bearer {self.auth_token}"},
                json=payload,
                timeout=self.request_timeout_s,
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
        except httpx.HTTPError as exc:
            raise RuntimeError(f"{method} {url} failed: {exc}") from exc


class WorkerApiClient(_ExecutionCoordinatorApiClientBase):
    def lease_job(self, *, resources: ResourceRequest) -> LeaseJobResponse:
        response = self._request_json(
            "/worker/lease_job",
            method="POST",
            payload=LeaseJobRequest(
                resources=resources, worker=_CURRENT_COMPONENT.get()
            ).model_dump(mode="json", exclude_none=True),
        )
        return TypeAdapter(LeaseJobResponse).validate_python(response)

    def job_result(self, lease_id: str, request: JobResultRequest) -> None:
        response = self._request_json(
            f"/worker/job_result/{lease_id}",
            method="POST",
            payload=request.model_dump(mode="json"),
        )
        OkResponse.model_validate(response)


class PoolApiClient(_ExecutionCoordinatorApiClientBase):
    def count_satisfiable_jobs(
        self,
        *,
        resources: ResourceRequest,
        max_workers: int,
        lost_workers: Sequence[str] = (),
    ) -> int:
        response = self._request_json(
            "/pool/count_satisfiable_jobs",
            method="POST",
            payload=CountSatisfiableJobsRequest(
                resources=resources,
                max_workers=max_workers,
                lost_workers=tuple(lost_workers),
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


def create_execution_coordinator_api_app(
    coordinator: ExecutionCoordinator, *, auth_token: str
) -> FastAPI:
    def require_auth(authorization: str = Header(default="")) -> None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not compare_digest(token, auth_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid furu execution coordinator auth token",
            )

    app = FastAPI()
    auth_dependency = Depends(require_auth)

    worker_router = APIRouter(prefix="/worker", dependencies=[auth_dependency])

    @worker_router.post("/lease_job", response_model=LeaseJobResponse)
    def lease_job(request: LeaseJobRequest) -> LeaseJobResponse:
        return coordinator.lease_job(resources=request.resources, worker=request.worker)

    @worker_router.post("/job_result/{lease_id}", response_model=OkResponse)
    def job_result(lease_id: str, request: JobResultRequest) -> OkResponse:
        coordinator.job_result(lease_id, request)
        return OkResponse()

    pool_router = APIRouter(prefix="/pool", dependencies=[auth_dependency])

    @pool_router.post("/count_satisfiable_jobs")
    def count_satisfiable_jobs(request: CountSatisfiableJobsRequest) -> int:
        return coordinator.count_satisfiable_jobs(
            resources=request.resources,
            max_workers=request.max_workers,
            lost_workers=request.lost_workers,
        )

    @pool_router.post("/fail", response_model=OkResponse)
    def fail(request: FailRequest) -> OkResponse:
        coordinator.fail(request.message)
        return OkResponse()

    app.include_router(worker_router)
    app.include_router(pool_router)
    return app
