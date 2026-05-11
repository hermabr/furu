from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

from furu.graph import DiscoveredGraph, node_key_for
from furu.server import (
    CreateSubmissionRequest,
    Launcher,
    LeaseRecord,
    Scheduler,
    SchedulerClient,
    build_app,
    build_test_client,
)
from furu.submission import Submission

if TYPE_CHECKING:
    from furu.core import Furu


@runtime_checkable
class Executor(Protocol):
    def submit[T](
        self,
        *,
        roots: Sequence[Furu[T]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[T]: ...


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def _run_local_worker(
    *,
    lease_id: str,
    http_client: httpx.Client,
) -> None:
    from furu.server import LeaseResultEnvelope
    from furu.worker_execution import execute_one_artifact

    client = SchedulerClient(http_client=http_client, owns_client=False)
    lease_response = client.get_lease(lease_id)
    result = execute_one_artifact(lease_response.node)

    envelope = LeaseResultEnvelope(result=result)
    response = http_client.post(
        f"/api/v1/leases/{lease_id}/result",
        content=envelope.model_dump_json(),
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()


class _LocalLauncher(Launcher):
    def __init__(
        self,
        *,
        num_workers: int,
        http_client: httpx.Client,
    ) -> None:
        self._num_workers = num_workers
        self._http_client = http_client
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._shutdown = False

    def capacity(self) -> int:
        with self._lock:
            if self._shutdown:
                return 0
            return self._num_workers - len(self._active)

    def launch(self, leases: list[LeaseRecord]) -> None:
        with self._lock:
            if self._shutdown:
                return
            for lease in leases:
                self._active.add(lease.lease_id)

        for lease in leases:
            thread = threading.Thread(
                target=self._worker_main,
                args=(lease.lease_id,),
                name=f"furu-local-worker:{lease.lease_id[:8]}",
                daemon=True,
            )
            with self._lock:
                self._threads.append(thread)
            thread.start()

    def _worker_main(self, lease_id: str) -> None:
        try:
            _run_local_worker(
                lease_id=lease_id,
                http_client=self._http_client,
            )
        finally:
            with self._lock:
                self._active.discard(lease_id)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            threads = list(self._threads)

        for thread in threads:
            thread.join()


@dataclass(frozen=True, kw_only=True)
class LocalExecutor:
    num_workers: int = 1

    def submit[T](
        self,
        *,
        roots: Sequence[Furu[T]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[T]:
        token = _generate_token()

        deferred = _DeferredLauncher()
        scheduler = Scheduler(launcher=deferred)
        app = build_app(scheduler=scheduler, token=token)
        http_client = build_test_client(app=app, token=token)

        real_launcher = _LocalLauncher(
            num_workers=self.num_workers,
            http_client=http_client,
        )
        deferred.target = real_launcher
        scheduler.launcher = real_launcher

        client = SchedulerClient(http_client=http_client, owns_client=False)

        input_order = tuple(node_key_for(root) for root in roots)
        request = CreateSubmissionRequest(
            graph=graph,
            roots=graph.roots,
            input_order=input_order,
            single_input=single_input,
        )

        response = client.create_submission(request)

        def _on_done() -> None:
            real_launcher.shutdown()

        return Submission[T](
            submission_id=response.submission_id,
            roots=tuple(roots),
            single_input=single_input,
            client=client,
            on_done=_on_done,
        )


class _DeferredLauncher(Launcher):
    target: Launcher | None = None

    def capacity(self) -> int:
        if self.target is None:
            return 0
        return self.target.capacity()

    def launch(self, leases: list[LeaseRecord]) -> None:
        if self.target is None:
            return
        self.target.launch(leases)

    def shutdown(self) -> None:
        if self.target is None:
            return
        self.target.shutdown()


@dataclass
class _SlurmLauncher(Launcher):
    workers: int
    token: str
    server_url: str
    partition: str | None = None
    account: str | None = None
    time: str | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    gpus: int | None = None
    extra_sbatch_args: tuple[str, ...] = ()
    scheduler: Scheduler | None = None
    _submitted_jobs: list[str] = field(default_factory=list)

    def capacity(self) -> int:
        return self.workers

    def launch(self, leases: list[LeaseRecord]) -> None:
        if not leases:
            return
        if self.scheduler is None:
            raise RuntimeError("SLURM launcher missing scheduler reference")
        if shutil.which("sbatch") is None:
            raise RuntimeError(
                "sbatch is not available on PATH; SlurmExecutor requires SLURM"
            )

        job_group_id = self.scheduler.create_job_group(leases)

        sbatch_args: list[str] = ["sbatch", f"--array=0-{len(leases) - 1}"]

        if self.partition is not None:
            sbatch_args.append(f"--partition={self.partition}")
        if self.account is not None:
            sbatch_args.append(f"--account={self.account}")
        if self.time is not None:
            sbatch_args.append(f"--time={self.time}")
        if self.cpus_per_task is not None:
            sbatch_args.append(f"--cpus-per-task={self.cpus_per_task}")
        if self.mem is not None:
            sbatch_args.append(f"--mem={self.mem}")
        if self.gpus is not None:
            sbatch_args.append(f"--gpus={self.gpus}")

        sbatch_args.extend(self.extra_sbatch_args)

        wrap_cmd = (
            "python -m furu.worker"
            f" --server-url {self.server_url}"
            f" --token {self.token}"
            f" --job-group-id {job_group_id}"
        )
        sbatch_args.extend(["--wrap", wrap_cmd])

        env = dict(os.environ)
        env["FURU_SERVER_URL"] = self.server_url
        env["FURU_SERVER_TOKEN"] = self.token
        env["FURU_JOB_GROUP_ID"] = job_group_id

        result = subprocess.run(
            sbatch_args,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self._submitted_jobs.append(result.stdout.strip())

    def shutdown(self) -> None:
        pass


@dataclass(frozen=True, kw_only=True)
class SlurmExecutor:
    workers: int

    partition: str | None = None
    account: str | None = None
    time: str | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    gpus: int | None = None

    extra_sbatch_args: tuple[str, ...] = ()

    server_host: str | None = None
    server_port: int = 0
    server_url: str | None = None

    def submit[T](
        self,
        *,
        roots: Sequence[Furu[T]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[T]:
        import uvicorn

        token = _generate_token()

        host = self.server_host or "0.0.0.0"

        launcher = _SlurmLauncher(
            workers=self.workers,
            token=token,
            server_url="",
            partition=self.partition,
            account=self.account,
            time=self.time,
            cpus_per_task=self.cpus_per_task,
            mem=self.mem,
            gpus=self.gpus,
            extra_sbatch_args=self.extra_sbatch_args,
        )

        scheduler = Scheduler(launcher=launcher)
        launcher.scheduler = scheduler
        app = build_app(scheduler=scheduler, token=token)

        config = uvicorn.Config(
            app=app,
            host=host,
            port=self.server_port,
            log_level="warning",
        )
        server = uvicorn.Server(config=config)

        thread = threading.Thread(
            target=server.run,
            name="furu-scheduler-server",
            daemon=True,
        )
        thread.start()

        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("scheduler server thread exited before startup")
            time.sleep(0.01)

        bound_port = self.server_port
        for socket_info in server.servers:
            for sock in socket_info.sockets:
                bound_port = sock.getsockname()[1]
                break
            break

        server_url = self.server_url or f"http://{host}:{bound_port}"
        launcher.server_url = server_url

        client = SchedulerClient.for_remote(base_url=server_url, token=token)

        input_order = tuple(node_key_for(root) for root in roots)
        request = CreateSubmissionRequest(
            graph=graph,
            roots=graph.roots,
            input_order=input_order,
            single_input=single_input,
        )

        response = client.create_submission(request)

        def _on_done() -> None:
            server.should_exit = True
            thread.join(timeout=5.0)

        return Submission[T](
            submission_id=response.submission_id,
            roots=tuple(roots),
            single_input=single_input,
            client=client,
            on_done=_on_done,
        )
