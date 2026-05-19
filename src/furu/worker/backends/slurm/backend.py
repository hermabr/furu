from __future__ import annotations

import secrets
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from furu.resources import ResourceRequest
from furu.utils import write_private_file
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.backends.slurm.resources import SlurmResources


@dataclass(frozen=True, slots=True)
class SlurmWorkerBackend:
    max_workers: int
    resources: SlurmResources
    resource_request: ResourceRequest
    worker_connect_host: str
    manager_listen_host: str = "0.0.0.0"
    job_name: str = "furu-worker"
    poll_interval: float = 10.0

    def start_pool(
        self,
        *,
        server_url: str,
        auth_token: str,
        executor_dir: Path,
    ) -> SlurmWorkerPool:
        from furu.execution.api import ManagerApiClient

        client = ManagerApiClient(server_url, auth_token=auth_token)
        n_workers = client.count_satisfiable_jobs(
            resources=self.resource_request, max_workers=self.max_workers
        )

        scheme, rest = server_url.split("://", maxsplit=1)
        server_url = (
            f"{scheme}://{self.worker_connect_host}:{rest.rsplit(':', maxsplit=1)[1]}"
        )

        chdir = Path.cwd().resolve()
        worker_dir = executor_dir.resolve() / "workers"
        worker_dir.mkdir(parents=True, exist_ok=True)

        token_file = worker_dir / f"worker-{secrets.token_hex(16)}.token"
        write_private_file(token_file, auth_token, mode=0o600)

        array_job_id = self._launch_jobs(
            chdir=chdir,
            token_file=token_file,
            worker_dir=worker_dir,
            server_url=server_url,
            n_workers=n_workers,
        )

        return SlurmWorkerPool(
            array_job_id=array_job_id,
            n_workers=n_workers,
            poll_interval=self.poll_interval,
        )

    def _launch_jobs(
        self,
        chdir: Path,
        token_file: Path,
        worker_dir: Path,
        server_url: str,
        n_workers: int,
    ) -> str:
        script_path = self._write_sbatch_script(
            worker_dir=worker_dir, token_file=token_file, server_url=server_url
        )

        log_dir = worker_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--chdir={chdir}",
                f"--output={log_dir / 'furu-worker-%A-%a.out'}",
                f"--error={log_dir / 'furu-worker-%A-%a.err'}",
                f"--job-name={self.job_name}",
                f"--array=0-{n_workers - 1}",
                *self.resources.to_sbatch_args(),
                "--export=NIL",
                str(script_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        array_job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        return array_job_id

    def _write_sbatch_script(
        self, *, worker_dir: Path, token_file: Path, server_url: str
    ) -> Path:
        scripts_dir = worker_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"worker-{secrets.token_hex(16)}.sh"
        write_private_file(
            script_path,
            (
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "\n"
                f"exec {shlex.quote(sys.executable)} -m furu.worker.cli \\\n"
                f"    --server-url {shlex.quote(server_url)} \\\n"
                f"    --auth-token-file {shlex.quote(str(token_file))}\n"
            ),
            mode=0o700,
        )
        return script_path
