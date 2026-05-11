from __future__ import annotations

import os
import secrets
import shlex
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import uvicorn

from furu.graph import GraphFragment, NodeKey
from furu.server.app import create_app
from furu.server.client import SchedulerClient
from furu.server.models import LeaseResponse, SubmissionState
from furu.server.scheduler import SchedulerState
from furu.submission import Submission


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

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

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
        server = _UvicornThread(app)
        server.start()

        client = SchedulerClient(base_url=server.url, token=token)
        response = client.create_submission(
            graph=graph,
            roots=tuple(roots),
            input_order=tuple(input_order),
            single_input=single_input,
        )
        if (
            client.get_submission(response.submission_id).state
            == SubmissionState.RUNNING
            and shutil.which("sbatch") is None
        ):
            server.stop()
            raise RuntimeError("SlurmExecutor requires sbatch on PATH")

        manager = _SlurmWorkerManager(
            client=client,
            submission_id=response.submission_id,
            server_url=server.url,
            token=token,
            executor=self,
        )
        manager.start()
        return Submission(
            id=response.submission_id,
            _client=client,
            _input_order=tuple(input_order),
            _single_input=single_input,
            _cancel_callback=manager.cancel,
        )


class _UvicornThread:
    def __init__(self, app) -> None:
        bind_host = os.environ.get("FURU_SERVER_BIND_HOST", "127.0.0.1")
        port = int(os.environ.get("FURU_SERVER_PORT", _free_port(bind_host)))
        public_host = os.environ.get(
            "FURU_SERVER_PUBLIC_HOST",
            bind_host if bind_host != "0.0.0.0" else socket.getfqdn(),
        )
        self.url = f"http://{public_host}:{port}"
        config = uvicorn.Config(
            app,
            host=bind_host,
            port=port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="furu-slurm-scheduler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while not self._server.started:
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out starting Furu scheduler server")
            time.sleep(0.01)

    def stop(self) -> None:
        self._server.should_exit = True


class _SlurmWorkerManager:
    def __init__(
        self,
        *,
        client: SchedulerClient,
        submission_id: str,
        server_url: str,
        token: str,
        executor: SlurmExecutor,
    ) -> None:
        self._client = client
        self._submission_id = submission_id
        self._server_url = server_url
        self._token = token
        self._executor = executor
        self._job_ids: list[str] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"furu-slurm-manager:{submission_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._stop_event.set()
        if self._job_ids and shutil.which("scancel") is not None:
            subprocess.run(["scancel", *self._job_ids], check=False)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            status = self._client.get_submission(self._submission_id)
            if status.state != SubmissionState.RUNNING:
                return

            leases = self._client.reserve_leases(
                submission_id=self._submission_id,
                max_count=self._executor.workers,
            ).leases
            if leases:
                job_group = self._client.create_job_group(
                    [lease.lease_id for lease in leases]
                )
                self._job_ids.append(
                    self._submit_array(
                        leases=leases,
                        job_group_id=job_group.job_group_id,
                    )
                )
            else:
                time.sleep(0.5)

    def _submit_array(
        self,
        *,
        leases: Sequence[LeaseResponse],
        job_group_id: str,
    ) -> str:
        array_max = len(leases) - 1
        command = (
            "python -m furu.worker "
            '--server-url "$FURU_SERVER_URL" '
            '--token "$FURU_SERVER_TOKEN" '
            '--job-group-id "$FURU_JOB_GROUP_ID" '
            '--array-index "$SLURM_ARRAY_TASK_ID"'
        )
        export = ",".join(
            [
                "ALL",
                f"FURU_SERVER_URL={self._server_url}",
                f"FURU_SERVER_TOKEN={self._token}",
                f"FURU_JOB_GROUP_ID={job_group_id}",
            ]
        )
        args = [
            "sbatch",
            "--parsable",
            f"--array=0-{array_max}",
            f"--export={export}",
            f"--wrap={command}",
            *self._sbatch_resource_args(),
            *self._executor.extra_sbatch_args,
        ]
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
        if not job_id:
            raise RuntimeError(
                "sbatch did not return a job id: "
                + shlex.join(args)
                + f"\nstdout={completed.stdout!r}\nstderr={completed.stderr!r}"
            )
        return job_id

    def _sbatch_resource_args(self) -> tuple[str, ...]:
        args: list[str] = []
        if self._executor.partition is not None:
            args.append(f"--partition={self._executor.partition}")
        if self._executor.account is not None:
            args.append(f"--account={self._executor.account}")
        if self._executor.time is not None:
            args.append(f"--time={self._executor.time}")
        if self._executor.cpus_per_task is not None:
            args.append(f"--cpus-per-task={self._executor.cpus_per_task}")
        if self._executor.mem is not None:
            args.append(f"--mem={self._executor.mem}")
        if self._executor.gpus is not None:
            args.append(f"--gres=gpu:{self._executor.gpus}")
        return tuple(args)


def _free_port(host: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return str(sock.getsockname()[1])
