from __future__ import annotations

import json
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from furu.core import Furu
from furu.execution import _load_or_create_local
from furu.metadata import ArtifactSpec
from furu.worker.context import _DependencyNotReady, worker_execution_context
from furu.worker.protocol import BlockedRequest, FinishRequest, Job


class ServerUnavailable(RuntimeError):
    pass


def worker_loop(
    *,
    server_url: str,
    unavailable_timeout: float = 30,
    retry_interval: float = 5,
    wait_interval: float = 0.1,
) -> None:
    server_url = server_url.rstrip("/")

    while True:
        try:
            response = _request_json(
                f"{server_url}/get_job",
                unavailable_timeout=unavailable_timeout,
                retry_interval=retry_interval,
            )
        except ServerUnavailable:
            return

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
            try:
                _post_json(
                    f"{server_url}/blocked/{_quote_path(job.lease_id)}",
                    BlockedRequest(dependencies=dependencies),
                    unavailable_timeout=unavailable_timeout,
                    retry_interval=retry_interval,
                )
            except ServerUnavailable:
                return
        except BaseException as exc:
            try:
                _post_json(
                    f"{server_url}/finish/{_quote_path(job.lease_id)}",
                    FinishRequest(
                        status="failed",
                        error="".join(
                            traceback.format_exception(
                                type(exc),
                                exc,
                                exc.__traceback__,
                            )
                        ),
                    ),
                    unavailable_timeout=unavailable_timeout,
                    retry_interval=retry_interval,
                )
            except ServerUnavailable:
                return
        else:
            try:
                _post_json(
                    f"{server_url}/finish/{_quote_path(job.lease_id)}",
                    FinishRequest(status="completed"),
                    unavailable_timeout=unavailable_timeout,
                    retry_interval=retry_interval,
                )
            except ServerUnavailable:
                return


def _run_job(job: Job) -> None:
    obj = Furu.from_artifact(job.artifact)
    with worker_execution_context(lease_id=job.lease_id):
        _load_or_create_local(obj)


def _post_json(
    url: str,
    payload: FinishRequest | BlockedRequest,
    *,
    unavailable_timeout: float,
    retry_interval: float,
) -> Any:
    return _request_json(
        url,
        method="POST",
        payload=payload.model_dump(mode="json"),
        unavailable_timeout=unavailable_timeout,
        retry_interval=retry_interval,
    )


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: object | None = None,
    unavailable_timeout: float,
    retry_interval: float,
) -> Any:
    deadline = time.monotonic() + unavailable_timeout
    while True:
        try:
            request = _build_request(url, method=method, payload=payload)
            with urllib.request.urlopen(request, timeout=retry_interval) as response:
                body = response.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}")
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            now = time.monotonic()
            if now >= deadline:
                raise ServerUnavailable(
                    f"{method} {url} unavailable for {unavailable_timeout:g} seconds"
                ) from exc
            time.sleep(min(retry_interval, deadline - now))


def _build_request(
    url: str,
    *,
    method: str,
    payload: object | None,
) -> urllib.request.Request:
    headers: dict[str, str] = {}
    data: bytes | None = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def _quote_path(value: str) -> str:
    return urllib.parse.quote(value, safe="")
