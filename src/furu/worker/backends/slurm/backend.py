from __future__ import annotations

import secrets
import shlex
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from furu.config import _WORKER_JSON_CONFIG_FILE_ENV_VAR, get_config
from furu.execution.api import PoolApiClient
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
    worker_idle_timeout: float = field(
        default_factory=lambda: get_config().worker_idle_timeout_seconds
    )

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

        config = get_config()
        config_file = worker_dir / f"worker-{secrets.token_hex(16)}.config.json"
        write_private_file(
            config_file,
            config.model_dump_json(indent=2) + "\n",
            mode=0o600,
        )

        resource_request = ResourceRequest(
            cpus=self.resources.cpus_per_worker,
            gpus=self.resources.gpus,
        )

        scripts_dir = worker_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"worker-{secrets.token_hex(16)}.sh"
        write_private_file(
            script_path,
            (
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "\n"
                "export "
                f"{_WORKER_JSON_CONFIG_FILE_ENV_VAR}={shlex.quote(str(config_file))}\n"
                "\n"
                f"exec {shlex.quote(sys.executable)} -m furu.worker._cli \\\n"
                f"    --server-url {shlex.quote(server_url)} \\\n"
                f"    --auth-token-file {shlex.quote(str(token_file))} \\\n"
                f"    --idle-timeout {self.worker_idle_timeout} \\\n"
                f"    --resource-cpus {resource_request.cpus} \\\n"
                f"    --resource-gpus {resource_request.gpus}\n"
            ),
            mode=0o700,
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

        pool_holder: list[SlurmWorkerPool] = []
        pool = SlurmWorkerPool(
            _sbatch_base_args=sbatch_base_args,
            _script_path=script_path,
            _max_workers=self.max_workers,
            _resource_request=resource_request,
            _server_url=server_url,
            _auth_token=auth_token,
            _poll_interval=self.poll_interval,
            _client=PoolApiClient(server_url=server_url, auth_token=auth_token),
            _stop_event=threading.Event(),
            _scale_thread=threading.Thread(
                target=lambda: pool_holder[0]._scale_loop(),
                name="furu-slurm-worker-pool-scale",
            ),
            _job_ids=[],
        )
        pool_holder.append(pool)
        pool._scale_thread.start()
        return pool
