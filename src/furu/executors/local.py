from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from furu.graph import GraphFragment, NodeKey
from furu.server.app import create_app
from furu.server.client import SchedulerClient
from furu.server.models import SubmissionState
from furu.server.scheduler import SchedulerState
from furu.submission import Submission
from furu.worker_execution import WorkerExecutionResultKind, execute_one_artifact


@dataclass(frozen=True, kw_only=True)
class LocalExecutor:
    num_workers: int = 1

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")

    def submit(
        self,
        *,
        graph: GraphFragment,
        roots: Sequence[NodeKey],
        input_order: Sequence[NodeKey],
        single_input: bool,
    ) -> Submission[Any]:
        state = SchedulerState()
        token = secrets.token_urlsafe(32)
        app = create_app(state=state, token=token)
        client = SchedulerClient(app=app, token=token)
        response = client.create_submission(
            graph=graph,
            roots=tuple(roots),
            input_order=tuple(input_order),
            single_input=single_input,
        )
        manager = _LocalWorkerManager(
            app=app,
            token=token,
            submission_id=response.submission_id,
            num_workers=self.num_workers,
        )
        manager.start()
        return Submission(
            id=response.submission_id,
            _client=client,
            _input_order=tuple(input_order),
            _single_input=single_input,
            _cancel_callback=manager.stop,
        )


class _LocalWorkerManager:
    def __init__(
        self,
        *,
        app: FastAPI,
        token: str,
        submission_id: str,
        num_workers: int,
    ) -> None:
        self._app = app
        self._token = token
        self._submission_id = submission_id
        self._num_workers = num_workers
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="furu-local-worker",
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"furu-local-manager:{submission_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self) -> None:
        client = SchedulerClient(app=self._app, token=self._token)
        active: set[Future[None]] = set()

        try:
            while not self._stop_event.is_set():
                done = {future for future in active if future.done()}
                for future in done:
                    active.remove(future)
                    future.result()

                status = client.get_submission(self._submission_id)
                if status.state != SubmissionState.RUNNING and not active:
                    return

                available = self._num_workers - len(active)
                if status.state == SubmissionState.RUNNING and available > 0:
                    leases = client.reserve_leases(
                        submission_id=self._submission_id,
                        max_count=available,
                    ).leases
                    for lease in leases:
                        active.add(
                            self._executor.submit(
                                _run_one_local_worker,
                                app=self._app,
                                token=self._token,
                                lease_id=lease.lease_id,
                            )
                        )

                if not done:
                    time.sleep(0.02)
        finally:
            client.close()


def _run_one_local_worker(
    *,
    app: FastAPI,
    token: str,
    lease_id: str,
) -> None:
    client = SchedulerClient(app=app, token=token)
    try:
        lease = client.get_lease(lease_id)
        result = execute_one_artifact(
            lease_id=lease.lease_id,
            node_key=lease.node_key,
            artifact=lease.artifact,
        )

        match result.kind:
            case WorkerExecutionResultKind.DONE:
                client.complete(
                    lease.lease_id,
                    node_key=result.node_key,
                )
            case WorkerExecutionResultKind.DEPENDENCY_NOT_READY:
                if result.call_kind is None or result.graph_fragment is None:
                    raise RuntimeError(
                        "dependency result is missing dependency payload"
                    )
                client.report_dependency(
                    lease.lease_id,
                    blocked=result.node_key,
                    call_kind=result.call_kind,
                    dependencies=result.dependencies,
                    graph_fragment=result.graph_fragment,
                )
            case WorkerExecutionResultKind.FAILED:
                client.fail(
                    lease.lease_id,
                    node_key=result.node_key,
                    error_type=result.error_type or "UnknownError",
                    error_message=result.error_message or "",
                    traceback=result.traceback or "",
                )
    finally:
        client.close()
