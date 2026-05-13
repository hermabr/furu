from __future__ import annotations

import time
import traceback
import urllib.parse
from typing import Any

import httpx

from furu.core import Furu
from furu.execution import _execute_one
from furu.metadata import ArtifactSpec
from furu.worker.context import _DependencyNotReady, worker_execution_context
from furu.worker.protocol import (
    BlockedRequest,
    FinishFailedRequest,
    FinishRequest,
    FinishSuccessRequest,
    Job,
)


def worker_loop(
    *,
    server_url: str,
    wait_interval: float = 0.1,
) -> None:
    server_url = server_url.rstrip("/")

    while True:
        response = _request_json(f"{server_url}/get_job")

        if response == "stop":
            return
        if response == "wait":
            time.sleep(wait_interval)
            continue

        job = Job.model_validate(response)
        try:
            _run_job(job)
        except _DependencyNotReady as exc:
            dependencies = [ArtifactSpec.from_furu(dep) for dep in exc.dependencies]
            _post_json(
                f"{server_url}/blocked/{_quote_path(job.lease_id)}",
                BlockedRequest(dependencies=dependencies),
            )
        except Exception as exc:
            _post_json(
                f"{server_url}/finish/{_quote_path(job.lease_id)}",
                FinishFailedRequest(
                    error="".join(
                        traceback.format_exception(
                            type(exc),
                            exc,
                            exc.__traceback__,
                        )
                    ),
                ),
            )
        else:
            _post_json(
                f"{server_url}/finish/{_quote_path(job.lease_id)}",
                FinishSuccessRequest(),
            )


def _run_job(job: Job) -> None:
    obj = Furu.from_artifact(job.artifact)
    with worker_execution_context(lease_id=job.lease_id):
        _execute_one(obj)


def _post_json(
    url: str,
    payload: FinishRequest | BlockedRequest,
) -> Any:
    return _request_json(
        url,
        method="POST",
        payload=payload.model_dump(mode="json"),
    )


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: object | None = None,
) -> Any:
    try:
        response = httpx.request(method, url, json=payload, timeout=10.0)
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"{method} {url} failed with HTTP "
            f"{exc.response.status_code}: {exc.response.text}"
        ) from exc


def _quote_path(value: str) -> str:
    return urllib.parse.quote(value, safe="")
