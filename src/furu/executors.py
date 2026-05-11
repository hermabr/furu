from __future__ import annotations

import secrets
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from furu.core import Furu
from furu.graph import DiscoveredGraph, node_key_for
from furu.server import LeaseRecord, Scheduler, SchedulerClient, create_app
from furu.submission import Submission


class Executor(Protocol):
    def submit(
        self,
        *,
        roots: Sequence[Furu[Any]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[Any]: ...


@dataclass(frozen=True, kw_only=True)
class LocalExecutor:
    num_workers: int = 1

    def submit(
        self,
        *,
        roots: Sequence[Furu[Any]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[Any]:
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        token = secrets.token_hex(16)
        launcher = _LocalLauncher(num_workers=self.num_workers, token=token)
        scheduler = Scheduler(launcher=launcher)
        app = create_app(scheduler=scheduler, token=token)
        launcher.app = app

        client = SchedulerClient(TestClient(app), token=token)
        roots_tuple = tuple(roots)
        root_keys = tuple(node_key_for(root) for root in roots_tuple)
        try:
            submission_id = client.create_submission(
                graph=graph,
                roots=graph.roots,
                input_order=root_keys,
                single_input=single_input,
            )
        except BaseException:
            client.close()
            raise
        return Submission(
            id=submission_id,
            _client=client,
            _roots=roots_tuple,
            _single_input=single_input,
        )


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

    def submit(
        self,
        *,
        roots: Sequence[Furu[Any]],
        graph: DiscoveredGraph,
        single_input: bool,
    ) -> Submission[Any]:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

        token = secrets.token_hex(16)
        launcher = _SlurmLauncher(executor=self, token=token)
        scheduler = Scheduler(launcher=launcher)
        launcher.scheduler = scheduler

        bind_host = self.server_host or "127.0.0.1"
        app = create_app(scheduler=scheduler, token=token)
        server = _UvicornServerHandle(
            app=app,
            host=bind_host,
            port=self.server_port,
        )
        server.start()

        public_url = self.server_url or f"http://{bind_host}:{server.port}"
        launcher.server_url = public_url
        client = SchedulerClient.for_url(public_url, token=token)

        roots_tuple = tuple(roots)
        root_keys = tuple(node_key_for(root) for root in roots_tuple)
        try:
            submission_id = client.create_submission(
                graph=graph,
                roots=graph.roots,
                input_order=root_keys,
                single_input=single_input,
            )
        except BaseException:
            client.close()
            server.stop()
            raise

        def close_server() -> None:
            client.close()
            server.stop()

        return Submission(
            id=submission_id,
            _client=client,
            _roots=roots_tuple,
            _single_input=single_input,
            _on_terminal=close_server,
        )


class _LocalLauncher:
    def __init__(self, *, num_workers: int, token: str) -> None:
        self.num_workers = num_workers
        self.token = token
        self.app: FastAPI | None = None
        self._threads: list[threading.Thread] = []

    def capacity(self) -> int:
        return self.num_workers

    def launch(self, leases: Sequence[LeaseRecord]) -> None:
        if self.app is None:
            raise RuntimeError("local launcher has no FastAPI app")
        for lease in leases:
            thread = threading.Thread(
                target=self._run_worker,
                args=(lease.lease_id,),
                name=f"furu-local-worker:{lease.lease_id[:8]}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _run_worker(self, lease_id: str) -> None:
        from furu.worker import run_worker_once

        if self.app is None:
            raise RuntimeError("local launcher has no FastAPI app")
        client = SchedulerClient(TestClient(self.app), token=self.token)
        try:
            run_worker_once(client, lease_id=lease_id)
        finally:
            client.close()


class _SlurmLauncher:
    def __init__(self, *, executor: SlurmExecutor, token: str) -> None:
        self.executor = executor
        self.token = token
        self.server_url: str | None = None
        self.scheduler: Scheduler | None = None

    def capacity(self) -> int:
        return self.executor.workers

    def launch(self, leases: Sequence[LeaseRecord]) -> None:
        if not leases:
            return
        if self.scheduler is None:
            raise RuntimeError("SLURM launcher has no scheduler")
        if self.server_url is None:
            raise RuntimeError("SLURM launcher has no server URL")

        job_group_id = self.scheduler.create_job_group(leases)
        subprocess.run(
            self._sbatch_command(
                job_group_id=job_group_id,
                array_size=len(leases),
            ),
            check=True,
        )

    def _sbatch_command(self, *, job_group_id: str, array_size: int) -> list[str]:
        executor = self.executor
        cmd = ["sbatch", f"--array=0-{array_size - 1}"]
        if executor.partition is not None:
            cmd.append(f"--partition={executor.partition}")
        if executor.account is not None:
            cmd.append(f"--account={executor.account}")
        if executor.time is not None:
            cmd.append(f"--time={executor.time}")
        if executor.cpus_per_task is not None:
            cmd.append(f"--cpus-per-task={executor.cpus_per_task}")
        if executor.mem is not None:
            cmd.append(f"--mem={executor.mem}")
        if executor.gpus is not None:
            cmd.append(f"--gpus={executor.gpus}")
        cmd.extend(executor.extra_sbatch_args)

        export = ",".join(
            [
                "ALL",
                f"FURU_SERVER_URL={self.server_url}",
                f"FURU_SERVER_TOKEN={self.token}",
                f"FURU_JOB_GROUP_ID={job_group_id}",
            ]
        )
        worker_cmd = (
            f"{shlex.quote(sys.executable)} -m furu.worker "
            '--server-url "$FURU_SERVER_URL" '
            '--token "$FURU_SERVER_TOKEN" '
            '--job-group-id "$FURU_JOB_GROUP_ID"'
        )
        cmd.extend(
            [
                f"--export={export}",
                "--wrap",
                worker_cmd,
            ]
        )
        return cmd


class _UvicornServerHandle:
    def __init__(self, *, app: FastAPI, host: str, port: int) -> None:
        self.host = host
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.listen()
        self.port = self._socket.getsockname()[1]
        config = uvicorn.Config(
            app,
            host=host,
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._socket]},
            name="furu-scheduler-server",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("scheduler server stopped before it started")
            if time.monotonic() > deadline:
                raise TimeoutError("scheduler server did not start")
            time.sleep(0.01)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)
