from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from secrets import token_urlsafe
from typing import assert_never

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, basic_auth, serve

from furu.config import _read_worker_json_config
from furu.execution.execution_coordinator import ExecutionCoordinator
from furu.logging import get_logger, log_detail
from furu.worker.backends.protocol import WorkerPool
from furu.worker.protocol import (
    CancelMessage,
    HelloMessage,
    PoolHandoff,
    ReconnectMessage,
    TakeoverAccepted,
    TakeoverReady,
    TakeoverRefused,
    TakeoverRequest,
    first_message_adapter,
    job_result_adapter,
    takeover_response_adapter,
)

_TAKEOVER_REPLY_TIMEOUT_S = 120.0

logger = get_logger()


@dataclass(frozen=True, slots=True)
class ExecutionCoordinatorServer:
    bound_host: str
    bound_port: int
    auth_token: str

    @property
    def server_url(self) -> str:
        return f"ws://{self.bound_host}:{self.bound_port}"


def _serve_takeover(
    coordinator: ExecutionCoordinator,
    connection: ServerConnection,
    request: TakeoverRequest,
    redirect_workers: Callable[[TakeoverReady], None],
) -> None:
    pools: dict[str, WorkerPool] | None = None
    with coordinator.lock:
        keys = [key for key in request.pool_keys if key in coordinator.pools]
        if not keys:
            refused = "no worker pool with a matching configuration"
        elif coordinator.taken_over_by is not None:
            refused = (
                f"already being taken over by exec={coordinator.taken_over_by[:5]}"
            )
        else:
            refused = None
            coordinator.taken_over_by = request.executor_id
            pools = {key: coordinator.pools[key] for key in keys}
            pool_count = len(coordinator.pools)
    if refused is not None:
        logger.warning(
            "refused takeover by exec=%s: %s", request.executor_id[:5], refused
        )
        connection.send(TakeoverRefused(reason=refused).model_dump_json())
        return
    assert pools is not None
    handoffs = {key: pool.handoff() for key, pool in pools.items()}
    logger.info(
        "handed off %d of %d pools to exec=%s",
        len(handoffs),
        pool_count,
        request.executor_id[:5],
    )
    try:
        connection.send(TakeoverAccepted(handoffs=handoffs).model_dump_json())
        with suppress(ConnectionClosed):
            ready = TakeoverReady.model_validate_json(connection.recv())
            redirect_workers(ready)
    finally:
        coordinator.fail(f"execution taken over by exec={request.executor_id[:5]}")


@contextmanager
def request_takeover(
    *,
    executor_id: str,
    source_id: str,
    url: str,
    pool_keys: Sequence[str],
) -> Iterator[dict[str, PoolHandoff]]:
    """Inherit matching pools, then send their published targets to the old workers."""
    try:
        connection = connect(url, max_size=None)
    except OSError as exc:
        raise RuntimeError(
            f"cannot reach exec={source_id[:5]}; is that coordinator still running?"
        ) from exc
    with connection:
        connection.send(
            TakeoverRequest(
                executor_id=executor_id, pool_keys=list(pool_keys)
            ).model_dump_json()
        )
        match takeover_response_adapter.validate_json(
            connection.recv(timeout=_TAKEOVER_REPLY_TIMEOUT_S)
        ):
            case TakeoverRefused(reason=reason):
                raise RuntimeError(
                    f"exec={source_id[:5]} refused the takeover: {reason}"
                )
            case TakeoverAccepted(handoffs=handoffs):
                yield handoffs
                # Read on the publishing side: compute nodes may still see old NFS files.
                targets = {}
                for handoff in handoffs.values():
                    for path in handoff.worker_files:
                        target_url, config = _read_worker_json_config(path)
                        targets[str(path)] = ReconnectMessage(
                            url=target_url, config=config
                        )
                connection.send(TakeoverReady(targets=targets).model_dump_json())
            case _ as unreachable:
                assert_never(unreachable)


def _serve_worker(
    coordinator: ExecutionCoordinator,
    connection: ServerConnection,
    hello: HelloMessage,
) -> None:
    with coordinator.log_context():
        worker = hello.worker
        logger.info(
            "worker connected · %s%s",
            worker,
            f" · running {hello.running[0].log_label}" if hello.running else "",
            extra=log_detail(worker=worker, backend=hello.backend),
        )
        try:
            if hello.running:
                if not coordinator.adopt(hello.running, worker=worker):
                    connection.send(CancelMessage().model_dump_json())
                result = job_result_adapter.validate_json(connection.recv())
                for artifact in hello.running:
                    coordinator.job_result(artifact.object_id, result)
            while True:
                job = coordinator.lease_job(resources=hello.resources, worker=worker)
                if job is None:
                    if coordinator.taken_over_by is not None:
                        coordinator.done.wait()  # Keep idle workers connected for the redirect.
                    return
                connection.send(job.model_dump_json())
                result = job_result_adapter.validate_json(connection.recv())
                for artifact in job.artifacts:
                    coordinator.job_result(artifact.object_id, result)
        except ConnectionClosed:
            logger.warning(
                "worker disconnected · %s",
                worker,
                extra=log_detail(worker=worker),
            )
        finally:
            coordinator.worker_lost(worker)


@contextmanager
def execution_coordinator_server(
    coordinator: ExecutionCoordinator, *, bind_host: str, port: int
) -> Iterator[ExecutionCoordinatorServer]:
    auth_token = token_urlsafe(32)
    connections: set[ServerConnection] = set()
    connections_changed = threading.Condition()
    workers: dict[ServerConnection, HelloMessage] = {}
    redirects: dict[str, ReconnectMessage] = {}

    def redirect_workers(ready: TakeoverReady) -> None:
        with connections_changed:
            redirects.update(ready.targets)
            inherited = tuple(workers.items())
        for worker_connection, hello in inherited:
            if target := redirects.get(str(hello.coordinator_file)):
                with suppress(ConnectionClosed):
                    worker_connection.send(target.model_dump_json())
                worker_connection.close()

    def handler(connection: ServerConnection) -> None:
        with connections_changed:
            connections.add(connection)
        try:
            with coordinator.log_context():
                first_message = first_message_adapter.validate_json(
                    connection.recv(timeout=10.0)
                )
                match first_message:
                    case HelloMessage() as hello:
                        with connections_changed:
                            workers[connection] = hello
                            target = redirects.get(str(hello.coordinator_file))
                        if target is not None:
                            connection.send(target.model_dump_json())
                        else:
                            _serve_worker(coordinator, connection, hello)
                    case TakeoverRequest() as request:
                        _serve_takeover(
                            coordinator, connection, request, redirect_workers
                        )
                    case _ as unreachable:
                        assert_never(unreachable)
        finally:
            with connections_changed:
                workers.pop(connection, None)
                connections.discard(connection)
                connections_changed.notify_all()

    server = serve(
        handler,
        bind_host,
        port,
        process_request=basic_auth(credentials=("furu", auth_token)),
        max_size=None,
    )
    bound_host, bound_port = server.socket.getsockname()[:2]
    thread = threading.Thread(
        target=server.serve_forever,
        name="furu-execution-coordinator-server",
    )
    thread.start()
    try:
        yield ExecutionCoordinatorServer(
            bound_host=bound_host,
            bound_port=bound_port,
            auth_token=auth_token,
        )
    finally:
        coordinator.fail("execution coordinator server closed before the run finished")
        server.shutdown()
        thread.join(timeout=10)
        with connections_changed:
            open_connections = tuple(connections)
        for connection in open_connections:
            connection.close()
        with connections_changed:
            connections_changed.wait_for(lambda: not connections, timeout=10)
