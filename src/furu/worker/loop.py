from __future__ import annotations

import threading
import time
from typing import Any

import httpx

from furu.execution.manager import GetJobResponse, Job
from furu.logging import get_logger
from furu.metadata import ArtifactSpec
from furu.serialize import load_furu_from_artifact
from furu.worker.context import _DependencyNotReady, worker_execution_context

_UNAVAILABLE_TOTAL_WAIT_SECONDS = 30.0
_UNAVAILABLE_RETRY_SECONDS = 5.0
_WAIT_POLL_SECONDS = 0.05


def worker_loop(base_url: str, *, worker_id: str) -> None:
    """Run a worker: pull jobs from the manager and execute them.

    Exits cleanly when the manager returns ``"stop"``. If the manager endpoint
    becomes unreachable, retries every 5 seconds for up to 30 seconds, then
    exits with an error log.
    """
    logger = get_logger(f"worker.{worker_id}")
    logger.debug("worker started against %s", base_url)
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        while True:
            try:
                response = _get_with_retry(client)
            except _ManagerUnavailable:
                logger.error(
                    "manager unavailable for >%.0fs; worker exiting",
                    _UNAVAILABLE_TOTAL_WAIT_SECONDS,
                )
                return

            if response == "stop":
                logger.debug("worker stopping (no more work)")
                return
            if response == "wait":
                time.sleep(_WAIT_POLL_SECONDS)
                continue

            job = response
            _run_job(client, job, logger=logger)


def spawn_workers(base_url: str, *, n_workers: int) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    for index in range(n_workers):
        worker_id = f"{index:02d}"
        thread = threading.Thread(
            target=worker_loop,
            kwargs={"base_url": base_url, "worker_id": worker_id},
            daemon=True,
            name=f"furu-worker-{worker_id}",
        )
        thread.start()
        threads.append(thread)
    return threads


class _ManagerUnavailable(Exception):
    pass


def _get_with_retry(client: httpx.Client) -> GetJobResponse:
    deadline = time.monotonic() + _UNAVAILABLE_TOTAL_WAIT_SECONDS
    last_error: Exception | None = None
    while True:
        try:
            response = client.get("/get_job")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise _ManagerUnavailable() from last_error
            time.sleep(_UNAVAILABLE_RETRY_SECONDS)
            continue
        return _parse_get_job(response.json())


def _parse_get_job(payload: Any) -> GetJobResponse:
    if payload == "wait":
        return "wait"
    if payload == "stop":
        return "stop"
    return Job.model_validate(payload)


def _run_job(client: httpx.Client, job: Job, *, logger: Any) -> None:
    obj = load_furu_from_artifact(job.artifact)
    logger.info(
        "leased %s as %s",
        obj._log_label,
        job.lease_id,
    )

    from furu.execution import _execute_one

    try:
        with worker_execution_context(lease_id=job.lease_id):
            _execute_one(obj)
    except _DependencyNotReady as exc:
        deps = [ArtifactSpec.from_furu(dep) for dep in exc.dependencies]
        logger.info(
            "lease %s reported %d missing dependency/dependencies",
            job.lease_id,
            len(deps),
        )
        response = client.post(
            f"/blocked/{job.lease_id}",
            json={"dependencies": [d.model_dump(mode="json") for d in deps]},
        )
        response.raise_for_status()
        return
    except BaseException:
        logger.exception("lease %s failed", job.lease_id)
        response = client.post(
            f"/finish/{job.lease_id}",
            json={"success": False},
        )
        response.raise_for_status()
        return

    response = client.post(
        f"/finish/{job.lease_id}",
        json={"success": True},
    )
    response.raise_for_status()
