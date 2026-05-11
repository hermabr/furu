from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import uvicorn

from furu.core import Furu
from furu.graph import DiscoveredGraph
from furu.server import (
    CreateSubmissionRequest,
    HttpSchedulerClient,
    LeaseRecord,
    Scheduler,
    SchedulerClient,
    TestSchedulerClient,
    create_app,
    new_token,
)
from furu.submission import Submission
from furu.worker_execution import execute_one_artifact


class Executor(Protocol):
    def submit[T](
        self,
        *,
        roots: Sequence[Furu[Any]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[T]: ...


def run_worker_for_lease(*, client: SchedulerClient, lease_id: str) -> None:
    lease = client.get_lease(lease_id)
    result = execute_one_artifact(lease)
    client.post_lease_result(lease_id=lease.lease_id, result=result)


class _LocalLauncher:
    def __init__(self, *, num_workers: int) -> None:
        self.num_workers = num_workers
        self.client: TestSchedulerClient | None = None
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def capacity(self) -> int:
        self._reap()
        return self.num_workers

    def launch(self, leases: list[LeaseRecord]) -> None:
        assert self.client is not None
        for lease in leases:
            thread = threading.Thread(
                target=run_worker_for_lease,
                kwargs={"client": self.client, "lease_id": lease.lease_id},
                daemon=True,
            )
            thread.start()
            with self._lock:
                self._threads.append(thread)

    def _reap(self) -> None:
        with self._lock:
            self._threads = [thread for thread in self._threads if thread.is_alive()]


@dataclass(frozen=True, kw_only=True)
class LocalExecutor:
    num_workers: int = 1

    def submit[T](
        self,
        *,
        roots: Sequence[Furu[Any]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[T]:
        token = new_token()
        launcher = _LocalLauncher(num_workers=self.num_workers)
        scheduler = Scheduler(launcher=launcher)
        app = create_app(scheduler=scheduler, token=token)
        client = TestSchedulerClient(app=app, token=token)
        launcher.client = client

        req = CreateSubmissionRequest(
            graph=graph,
            roots=graph.roots,
            input_order=graph.roots,
            single_input=single_input,
        )
        submission_id = client.create_submission(req)
        return Submission(
            id=submission_id,
            _client=client,
            _roots=roots,
            _single_input=single_input,
        )


class _SlurmLauncher:
    def __init__(
        self,
        *,
        workers: int,
        scheduler: Scheduler | None,
        server_url: str,
        token: str,
        executor: SlurmExecutor,
    ) -> None:
        self.workers = workers
        self.scheduler = scheduler
        self.server_url = server_url
        self.token = token
        self.executor = executor

    def capacity(self) -> int:
        return self.workers

    def launch(self, leases: list[LeaseRecord]) -> None:
        if not leases:
            return
        assert self.scheduler is not None
        job_group_id = self.scheduler.create_job_group(leases)
        array_spec = f"0-{len(leases) - 1}"
        command = ["sbatch", f"--array={array_spec}"]
        if self.executor.partition is not None:
            command.append(f"--partition={self.executor.partition}")
        if self.executor.account is not None:
            command.append(f"--account={self.executor.account}")
        if self.executor.time is not None:
            command.append(f"--time={self.executor.time}")
        if self.executor.cpus_per_task is not None:
            command.append(f"--cpus-per-task={self.executor.cpus_per_task}")
        if self.executor.mem is not None:
            command.append(f"--mem={self.executor.mem}")
        if self.executor.gpus is not None:
            command.append(f"--gpus={self.executor.gpus}")
        command.extend(self.executor.extra_sbatch_args)
        env = os.environ.copy()
        env.update(
            {
                "FURU_SERVER_URL": self.server_url,
                "FURU_SERVER_TOKEN": self.token,
                "FURU_JOB_GROUP_ID": job_group_id,
            }
        )
        worker = (
            'python -m furu.worker --server-url "$FURU_SERVER_URL" '
            '--token "$FURU_SERVER_TOKEN" --job-group-id "$FURU_JOB_GROUP_ID"'
        )
        command.extend(["--wrap", worker])
        subprocess.run(command, env=env, check=True)


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
        roots: Sequence[Furu[Any]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[T]:
        token = new_token()
        host = self.server_host or "127.0.0.1"
        scheduler = Scheduler(
            launcher=_SlurmLauncher(
                workers=self.workers,
                scheduler=None,
                server_url="",
                token=token,
                executor=self,
            )
        )
        launcher = scheduler.launcher
        assert isinstance(launcher, _SlurmLauncher)
        launcher.scheduler = scheduler
        app = create_app(scheduler=scheduler, token=token)
        config = uvicorn.Config(
            app,
            host=host,
            port=self.server_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        url = self.server_url or f"http://{host}:{port}"
        launcher.server_url = url
        client = HttpSchedulerClient(server_url=url, token=token)
        submission_id = client.create_submission(
            CreateSubmissionRequest(
                graph=graph,
                roots=graph.roots,
                input_order=graph.roots,
                single_input=single_input,
            )
        )

        def shutdown() -> None:
            server.should_exit = True
            client.close()

        return Submission(
            id=submission_id,
            _client=client,
            _roots=roots,
            _single_input=single_input,
            _on_terminal=shutdown,
        )
