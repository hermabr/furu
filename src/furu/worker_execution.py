from __future__ import annotations

import json
import time
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict

from furu.metadata import ArtifactSpec

if TYPE_CHECKING:
    from furu.core import Furu

type DependencyCallKind = Literal["load_or_create", "try_load"]


_worker_execution_lease_id: ContextVar[str | None] = ContextVar(
    "_worker_execution_lease_id",
    default=None,
)


@contextmanager
def worker_execution_context(
    *,
    lease_id: str,
) -> Iterator[None]:
    token = _worker_execution_lease_id.set(lease_id)

    try:
        yield
    finally:
        _worker_execution_lease_id.reset(token)


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lease_id: str
    artifact: ArtifactSpec


class FinishUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["completed", "failed"]
    error: str | None = None


class BlockedUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dependencies: list[ArtifactSpec]


class _EndpointUnavailable(RuntimeError):
    pass


def _request_json_once(
    *,
    method: str,
    url: str,
    payload: object | None = None,
    request_timeout_seconds: float,
) -> object:
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=request_timeout_seconds) as response:
        body = response.read()
    if not body:
        return None
    return json.loads(body)


def _request_json(
    *,
    method: str,
    url: str,
    payload: object | None = None,
    unavailable_timeout_seconds: float,
    unavailable_check_interval_seconds: float,
    request_timeout_seconds: float,
) -> object:
    deadline = time.monotonic() + unavailable_timeout_seconds
    while True:
        try:
            return _request_json_once(
                method=method,
                url=url,
                payload=payload,
                request_timeout_seconds=request_timeout_seconds,
            )
        except HTTPError:
            raise
        except (OSError, TimeoutError, URLError) as exc:
            if time.monotonic() >= deadline:
                raise _EndpointUnavailable(url) from exc
            time.sleep(unavailable_check_interval_seconds)


def _post_finish(
    *,
    base_url: str,
    lease_id: str,
    update: FinishUpdate,
    unavailable_timeout_seconds: float,
    unavailable_check_interval_seconds: float,
    request_timeout_seconds: float,
) -> bool:
    try:
        _request_json(
            method="POST",
            url=f"{base_url}/finish/{lease_id}",
            payload=update.model_dump(mode="json"),
            unavailable_timeout_seconds=unavailable_timeout_seconds,
            unavailable_check_interval_seconds=unavailable_check_interval_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
    except _EndpointUnavailable:
        return False
    return True


def _post_blocked(
    *,
    base_url: str,
    lease_id: str,
    update: BlockedUpdate,
    unavailable_timeout_seconds: float,
    unavailable_check_interval_seconds: float,
    request_timeout_seconds: float,
) -> bool:
    try:
        _request_json(
            method="POST",
            url=f"{base_url}/blocked/{lease_id}",
            payload=update.model_dump(mode="json"),
            unavailable_timeout_seconds=unavailable_timeout_seconds,
            unavailable_check_interval_seconds=unavailable_check_interval_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
    except _EndpointUnavailable:
        return False
    return True


def worker_loop(
    base_url: str,
    *,
    wait_sleep_seconds: float = 0.1,
    unavailable_timeout_seconds: float = 30.0,
    unavailable_check_interval_seconds: float = 5.0,
    request_timeout_seconds: float = 5.0,
) -> None:
    from furu.core import Furu
    from furu.execution import _load_or_create_local

    base_url = base_url.rstrip("/")

    while True:
        try:
            payload = _request_json(
                method="GET",
                url=f"{base_url}/get_job",
                unavailable_timeout_seconds=unavailable_timeout_seconds,
                unavailable_check_interval_seconds=unavailable_check_interval_seconds,
                request_timeout_seconds=request_timeout_seconds,
            )
        except _EndpointUnavailable:
            return

        if payload == "wait":
            time.sleep(wait_sleep_seconds)
            continue
        if payload == "stop":
            return

        job = Job.model_validate(payload)

        try:
            obj = Furu.from_artifact(job.artifact)
            with worker_execution_context(lease_id=job.lease_id):
                _load_or_create_local(obj)
        except _DependencyNotReady as exc:
            if not _post_blocked(
                base_url=base_url,
                lease_id=job.lease_id,
                update=BlockedUpdate(
                    dependencies=[
                        ArtifactSpec.from_furu(dep) for dep in exc.dependencies
                    ]
                ),
                unavailable_timeout_seconds=unavailable_timeout_seconds,
                unavailable_check_interval_seconds=unavailable_check_interval_seconds,
                request_timeout_seconds=request_timeout_seconds,
            ):
                return
        except BaseException as exc:
            if not _post_finish(
                base_url=base_url,
                lease_id=job.lease_id,
                update=FinishUpdate(
                    status="failed",
                    error="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                ),
                unavailable_timeout_seconds=unavailable_timeout_seconds,
                unavailable_check_interval_seconds=unavailable_check_interval_seconds,
                request_timeout_seconds=request_timeout_seconds,
            ):
                return
        else:
            if not _post_finish(
                base_url=base_url,
                lease_id=job.lease_id,
                update=FinishUpdate(status="completed"),
                unavailable_timeout_seconds=unavailable_timeout_seconds,
                unavailable_check_interval_seconds=unavailable_check_interval_seconds,
                request_timeout_seconds=request_timeout_seconds,
            ):
                return


class _DependencyNotReady(BaseException):
    dependencies: tuple[Furu[Any], ...]
    call_kind: DependencyCallKind

    def __init__(
        self,
        dependencies: Sequence[Furu[Any]],
        *,
        call_kind: DependencyCallKind,
    ) -> None:
        self.dependencies = tuple(dependencies)
        self.call_kind = call_kind

        super().__init__(
            f"{call_kind} discovered "
            f"{len(self.dependencies)} missing dependency/dependencies"
        )
