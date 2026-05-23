from __future__ import annotations

import secrets
import shlex
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
        scheme, rest = server_url.split("://", maxsplit=1)
        server_url = (
            f"{scheme}://{self.worker_connect_host}:{rest.rsplit(':', maxsplit=1)[1]}"
        )

        chdir = Path.cwd().resolve()
        worker_dir = executor_dir.resolve() / "workers"
        worker_dir.mkdir(parents=True, exist_ok=True)

        token_file = worker_dir / f"worker-{secrets.token_hex(16)}.token"
        write_private_file(token_file, auth_token, mode=0o600)

        resource_request = ResourceRequest(
            cpus=self.resources.cpus_per_worker,
            gpus=self.resources.gpus,
        )

        script_path = self._write_sbatch_script(
            worker_dir=worker_dir,
            token_file=token_file,
            server_url=server_url,
            resource_request=resource_request,
        )

        log_dir = worker_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        sbatch_base_args = (
            f"--chdir={chdir}",
            f"--output={log_dir / 'furu-worker-%j.out'}",
            f"--error={log_dir / 'furu-worker-%j.err'}",
            f"--job-name={self.job_name}",
            *self.resources.to_sbatch_args(),
            "--export=NIL",
        )

        return SlurmWorkerPool(
            sbatch_base_args=sbatch_base_args,
            script_path=script_path,
            max_workers=self.max_workers,
            resource_request=resource_request,
            server_url=server_url,
            auth_token=auth_token,
            poll_interval=self.poll_interval,
        )

    def _write_sbatch_script(
        self,
        *,
        worker_dir: Path,
        token_file: Path,
        server_url: str,
        resource_request: ResourceRequest,
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
                f"exec {shlex.quote(sys.executable)} -m furu.worker._cli \\\n"
                f"    --server-url {shlex.quote(server_url)} \\\n"
                f"    --auth-token-file {shlex.quote(str(token_file))} \\\n"
                f"    --resource-cpus {resource_request.cpus} \\\n"
                f"    --resource-gpus {resource_request.gpus}\n"
            ),
            mode=0o700,
        )
        return script_path
