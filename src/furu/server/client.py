from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from furu.graph import GraphFragment, NodeKey
from furu.server.models import (
    CompleteLeaseRequest,
    CreateJobGroupRequest,
    CreateJobGroupResponse,
    CreateSubmissionRequest,
    CreateSubmissionResponse,
    DependencyLeaseRequest,
    FailLeaseRequest,
    LeaseResponse,
    ReserveLeasesRequest,
    ReserveLeasesResponse,
    SubmissionGraphView,
    SubmissionState,
    SubmissionStatus,
)


class SchedulerClient:
    def __init__(
        self,
        *,
        token: str,
        base_url: str | None = None,
        app: FastAPI | None = None,
    ) -> None:
        if (base_url is None) == (app is None):
            raise ValueError("exactly one of base_url or app is required")

        self._headers = {"Authorization": f"Bearer {token}"}
        self._http_client: httpx.Client | None = None
        self._test_client: TestClient | None = None
        if app is None:
            assert base_url is not None
            self._http_client = httpx.Client(base_url=base_url, headers=self._headers)
        else:
            self._test_client = TestClient(app, headers=self._headers)

    def create_submission(
        self,
        *,
        graph: GraphFragment,
        roots: Sequence[NodeKey],
        input_order: Sequence[NodeKey],
        single_input: bool,
    ) -> CreateSubmissionResponse:
        request = CreateSubmissionRequest(
            graph=graph,
            roots=tuple(roots),
            input_order=tuple(input_order),
            single_input=single_input,
        )
        return self._request_model(
            "POST",
            "/api/v1/submissions",
            CreateSubmissionResponse,
            json=_model_json(request),
        )

    def get_submission(self, submission_id: str) -> SubmissionStatus:
        return self._request_model(
            "GET",
            f"/api/v1/submissions/{submission_id}",
            SubmissionStatus,
        )

    def get_graph(self, submission_id: str) -> SubmissionGraphView:
        return self._request_model(
            "GET",
            f"/api/v1/submissions/{submission_id}/graph",
            SubmissionGraphView,
        )

    def cancel_submission(self, submission_id: str) -> None:
        self._request(
            "POST",
            f"/api/v1/submissions/{submission_id}/cancel",
        )

    def reserve_leases(
        self,
        *,
        submission_id: str,
        max_count: int,
    ) -> ReserveLeasesResponse:
        request = ReserveLeasesRequest(max_count=max_count)
        return self._request_model(
            "POST",
            f"/api/v1/submissions/{submission_id}/leases/reserve",
            ReserveLeasesResponse,
            json=_model_json(request),
        )

    def get_lease(self, lease_id: str) -> LeaseResponse:
        return self._request_model(
            "GET",
            f"/api/v1/leases/{lease_id}",
            LeaseResponse,
        )

    def complete(self, lease_id: str, *, node_key: NodeKey) -> None:
        request = CompleteLeaseRequest(node_key=node_key)
        self._request(
            "POST",
            f"/api/v1/leases/{lease_id}/complete",
            json=_model_json(request),
        )

    def report_dependency(
        self,
        lease_id: str,
        *,
        blocked: NodeKey,
        call_kind: Literal["load_or_create", "try_load"],
        dependencies: Sequence[NodeKey],
        graph_fragment: GraphFragment,
    ) -> None:
        request = DependencyLeaseRequest(
            blocked=blocked,
            call_kind=call_kind,
            dependencies=tuple(dependencies),
            graph_fragment=graph_fragment,
        )
        self._request(
            "POST",
            f"/api/v1/leases/{lease_id}/dependency",
            json=_model_json(request),
        )

    def fail(
        self,
        lease_id: str,
        *,
        node_key: NodeKey,
        error_type: str,
        error_message: str,
        traceback: str,
    ) -> None:
        request = FailLeaseRequest(
            node_key=node_key,
            error_type=error_type,
            error_message=error_message,
            traceback=traceback,
        )
        self._request(
            "POST",
            f"/api/v1/leases/{lease_id}/fail",
            json=_model_json(request),
        )

    def create_job_group(self, lease_ids: Sequence[str]) -> CreateJobGroupResponse:
        request = CreateJobGroupRequest(lease_ids=tuple(lease_ids))
        return self._request_model(
            "POST",
            "/api/v1/job-groups",
            CreateJobGroupResponse,
            json=_model_json(request),
        )

    def get_lease_for_array_index(
        self,
        *,
        job_group_id: str,
        array_index: int,
    ) -> LeaseResponse:
        return self._request_model(
            "GET",
            f"/api/v1/job-groups/{job_group_id}/leases/{array_index}",
            LeaseResponse,
        )

    def wait_until_done(
        self,
        submission_id: str,
        *,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.05,
        on_poll: Callable[[], None] | None = None,
    ) -> SubmissionStatus:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if on_poll is not None:
                on_poll()
            status = self.get_submission(submission_id)
            if status.state != SubmissionState.RUNNING:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"submission {submission_id} did not finish")
            time.sleep(poll_interval_s)

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
        if self._test_client is not None:
            self._test_client.close()

    def _request_model[T: BaseModel](
        self,
        method: str,
        path: str,
        model: type[T],
        *,
        json: object | None = None,
    ) -> T:
        response = self._request(method, path, json=json)
        return model.model_validate_json(response.content)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
    ) -> httpx.Response:
        if self._http_client is not None:
            response = self._http_client.request(method, path, json=json)
        else:
            assert self._test_client is not None
            response = self._test_client.request(method, path, json=json)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Furu scheduler request failed: {response.status_code} {response.text}"
            )
        return response


def _model_json(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
