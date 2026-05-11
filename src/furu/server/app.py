from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from furu.server.models import (
    CompleteLeaseRequest,
    CreateJobGroupRequest,
    CreateJobGroupResponse,
    CreateSubmissionRequest,
    CreateSubmissionResponse,
    DependencyLeaseRequest,
    FailLeaseRequest,
    FailureRecord,
    LeaseResponse,
    ReserveLeasesRequest,
    ReserveLeasesResponse,
    SubmissionGraphView,
    SubmissionStatus,
)
from furu.server.scheduler import (
    ProtocolError,
    SchedulerState,
    lease_response_from_record,
)


def create_app(*, state: SchedulerState, token: str) -> FastAPI:
    app = FastAPI()

    def require_token(authorization: str = Header(default="")) -> None:
        expected = f"Bearer {token}"
        if authorization != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid FURU server token",
            )

    @app.exception_handler(ProtocolError)
    async def protocol_error_handler(
        _request: object,
        exc: ProtocolError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.post(
        "/api/v1/submissions",
        response_model=CreateSubmissionResponse,
        dependencies=[Depends(require_token)],
    )
    async def create_submission(
        http_request: Request,
    ) -> CreateSubmissionResponse:
        request = await _parse_strict_json_body(http_request, CreateSubmissionRequest)
        return CreateSubmissionResponse(
            submission_id=state.create_submission(request),
        )

    @app.get(
        "/api/v1/submissions/{submission_id}",
        response_model=SubmissionStatus,
        dependencies=[Depends(require_token)],
    )
    def get_submission(submission_id: str) -> SubmissionStatus:
        return state.get_submission(submission_id)

    @app.get(
        "/api/v1/submissions/{submission_id}/graph",
        response_model=SubmissionGraphView,
        dependencies=[Depends(require_token)],
    )
    def get_submission_graph(submission_id: str) -> SubmissionGraphView:
        return state.get_graph(submission_id)

    @app.post(
        "/api/v1/submissions/{submission_id}/cancel",
        dependencies=[Depends(require_token)],
    )
    def cancel_submission(submission_id: str) -> dict[str, bool]:
        state.cancel_submission(submission_id)
        return {"ok": True}

    @app.post(
        "/api/v1/submissions/{submission_id}/leases/reserve",
        response_model=ReserveLeasesResponse,
        dependencies=[Depends(require_token)],
    )
    def reserve_leases(
        submission_id: str,
        request: ReserveLeasesRequest,
    ) -> ReserveLeasesResponse:
        leases = state.reserve_ready_nodes(
            submission_id=submission_id,
            max_count=request.max_count,
        )
        return ReserveLeasesResponse(
            leases=tuple(lease_response_from_record(lease) for lease in leases),
        )

    @app.get(
        "/api/v1/leases/{lease_id}",
        response_model=LeaseResponse,
        dependencies=[Depends(require_token)],
    )
    def get_lease(lease_id: str) -> LeaseResponse:
        return lease_response_from_record(state.get_lease(lease_id))

    @app.post(
        "/api/v1/leases/{lease_id}/complete",
        dependencies=[Depends(require_token)],
    )
    def complete_lease(
        lease_id: str,
        request: CompleteLeaseRequest,
    ) -> dict[str, bool]:
        state.complete_lease(lease_id, node_key=request.node_key)
        return {"ok": True}

    @app.post(
        "/api/v1/leases/{lease_id}/dependency",
        dependencies=[Depends(require_token)],
    )
    async def report_dependency(
        lease_id: str,
        http_request: Request,
    ) -> dict[str, bool]:
        request = await _parse_strict_json_body(http_request, DependencyLeaseRequest)
        state.report_dependency(
            lease_id,
            blocked=request.blocked,
            dependencies=request.dependencies,
            graph_fragment=request.graph_fragment,
        )
        return {"ok": True}

    @app.post(
        "/api/v1/leases/{lease_id}/fail",
        dependencies=[Depends(require_token)],
    )
    def fail_lease(
        lease_id: str,
        request: FailLeaseRequest,
    ) -> dict[str, bool]:
        state.fail_lease(
            lease_id,
            failure=FailureRecord(
                node_key=request.node_key,
                error_type=request.error_type,
                error_message=request.error_message,
                traceback=request.traceback,
            ),
        )
        return {"ok": True}

    @app.post(
        "/api/v1/job-groups",
        response_model=CreateJobGroupResponse,
        dependencies=[Depends(require_token)],
    )
    async def create_job_group(http_request: Request) -> CreateJobGroupResponse:
        request = await _parse_strict_json_body(http_request, CreateJobGroupRequest)
        group = state.create_job_group(request.lease_ids)
        return CreateJobGroupResponse(job_group_id=group.id)

    @app.get(
        "/api/v1/job-groups/{job_group_id}/leases/{array_index}",
        response_model=LeaseResponse,
        dependencies=[Depends(require_token)],
    )
    def get_lease_for_array_index(
        job_group_id: str,
        array_index: int,
    ) -> LeaseResponse:
        return lease_response_from_record(
            state.get_lease_for_array_index(
                job_group_id=job_group_id,
                array_index=array_index,
            )
        )

    return app


async def _parse_strict_json_body[T: BaseModel](
    request: Request,
    model: type[T],
) -> T:
    body = await request.body()
    try:
        return model.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc
