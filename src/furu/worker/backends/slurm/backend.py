from __future__ import annotations

import dataclasses
import os
import secrets
import shlex
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, assert_never

from furu.config import (
    _WORKER_JSON_CONFIG_FILE_ENV_VAR,
    get_config,
)
from furu.provenance import SubmitProvenance
from furu.resources import ResourceFloor, ResourceRequest, resource_request_adapter
from furu.snapshot import snapshot_code
from furu.utils import (
    _hash_dict_deterministically,
    replace_private_file,
    write_private_file,
)
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.backends.slurm.resources import SlurmResources
from furu.worker.protocol import PoolHandoff, coordinator_url

if TYPE_CHECKING:
    from furu.execution.execution_coordinator import ExecutionCoordinator

type SlurmExport = Literal["NIL", "ALL"] | tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class SlurmWorkerBackend:
    max_workers: int
    resources: SlurmResources
    worker_connect_host: str = field(
        default_factory=lambda: get_config().worker.connect_host or socket.getfqdn()
    )
    worker_connect_port: int | None = None
    execution_coordinator_listen_host: str = "0.0.0.0"
    job_name: str = "furu-worker"
    poll_interval: float = 10.0
    worker_idle_timeout: float = field(
        default_factory=lambda: get_config().worker.idle_timeout_seconds
    )
    pre_worker_commands: tuple[str, ...] = ()
    export: SlurmExport = None
    use_job_arrays: bool = True
    reserve_for: ResourceFloor = field(default_factory=ResourceFloor)

    def __post_init__(self) -> None:
        if not get_config().provenance.snapshot:
            raise ValueError(
                "Slurm workers run from a code snapshot; "
                "set [tool.furu.provenance] snapshot = true"
            )

    @property
    def resource_request(self) -> ResourceRequest:
        return ResourceRequest(
            cpus=self.resources.cpus_per_worker,
            gpus=self.resources.gpus,
            memory_gib=self.resources.memory_gib,
            reserve_for=self.reserve_for,
        )

    @property
    def pool_key(self) -> str:
        export = list(self.export) if isinstance(self.export, tuple) else self.export
        return "slurm:" + _hash_dict_deterministically(
            {
                "sbatch": self.resources.to_sbatch_args(),
                "reserve_for": dataclasses.asdict(self.reserve_for),
                "pre_worker_commands": list(self.pre_worker_commands),
                "export": export,
                "use_job_arrays": self.use_job_arrays,
            }
        )

    def start_pool(
        self,
        *,
        coordinator: ExecutionCoordinator,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        provenance: SubmitProvenance,
        handoff: PoolHandoff | None,
    ) -> SlurmWorkerPool:
        connect_port = (
            bound_port if self.worker_connect_port is None else self.worker_connect_port
        )
        url = coordinator_url(
            host=self.worker_connect_host, port=connect_port, auth_token=auth_token
        )

        code = snapshot_code(provenance)
        subprocess.run(
            ["uv", "sync", "--frozen", "--project", str(code.project_root)],
            env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
            check=True,
        )
        worker_dir = executor_dir.resolve() / "workers" / secrets.token_hex(8)
        worker_dir.mkdir(parents=True)

        coordinator_file = worker_dir / "coordinator.url"
        write_private_file(coordinator_file, url + "\n", mode=0o600)
        coordinator_files = [coordinator_file]
        job_ids: list[str] = []
        if handoff is not None:
            for inherited_file in handoff.coordinator_files:
                replace_private_file(inherited_file, url + "\n", mode=0o600)
            coordinator_files.extend(handoff.coordinator_files)
            job_ids.extend(handoff.job_ids)

        config = get_config()
        config = config.model_copy(
            update={"directories": config.directories.anchored()}
        )
        config_file = worker_dir / "worker.config.json"
        write_private_file(
            config_file,
            config.model_dump_json(indent=2) + "\n",
            mode=0o600,
        )

        resource_request = self.resource_request
        resources_json = resource_request_adapter.dump_json(resource_request).decode()
        pre_worker_script = "".join(
            f"{command}\n" for command in self.pre_worker_commands
        )
        if pre_worker_script:
            pre_worker_script += "\n"

        script_path = worker_dir / "worker.sh"
        if self.use_job_arrays:
            component_line = 'furu_worker_component="slurm-worker-${SLURM_ARRAY_JOB_ID}a${SLURM_ARRAY_TASK_ID}"\n'
        else:
            component_line = 'furu_worker_component="slurm-worker-${SLURM_JOB_ID}"\n'

        write_private_file(
            script_path,
            (
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "\n"
                "export "
                f"{_WORKER_JSON_CONFIG_FILE_ENV_VAR}={shlex.quote(str(config_file))}\n"
                "\n"
                f"{component_line}"
                "\n"
                f"{pre_worker_script}"
                # Do not leak the submit environment into snapshot workers.
                "unset VIRTUAL_ENV\n"
                "\n"
                "exec uv run --frozen "
                f"--project {shlex.quote(str(code.project_root))} \\\n"
                "    python -m furu.worker._cli \\\n"
                f"    --coordinator-file {shlex.quote(str(coordinator_file))} \\\n"
                '    --component "${furu_worker_component}" \\\n'
                "    --backend slurm \\\n"
                f"    --idle-timeout {self.worker_idle_timeout} \\\n"
                f"    --resources {shlex.quote(resources_json)}\n"
            ),
            mode=0o700,
        )

        log_dir = worker_dir / "logs"
        log_dir.mkdir()
        log_name = "furu-worker-%A_%a" if self.use_job_arrays else "furu-worker-%j"

        export_sbatch_arg: tuple[str, ...]
        match self.export:
            case None | ():
                export_sbatch_arg = ()
            case "NIL" | "ALL":
                export_sbatch_arg = (f"--export={self.export}",)
            case (*names,):
                export_sbatch_arg = (f"--export={','.join(names)}",)
            case _:
                assert_never(self.export)

        sbatch_base_args = (
            f"--chdir={code.cwd}",
            f"--output={log_dir / f'{log_name}.out'}",
            f"--error={log_dir / f'{log_name}.err'}",
            f"--job-name={self.job_name}",
            *self.resources.to_sbatch_args(),
            *export_sbatch_arg,
        )

        pool_holder: list[SlurmWorkerPool] = []
        pool = SlurmWorkerPool(
            _sbatch_base_args=sbatch_base_args,
            _script_path=script_path,
            _max_workers=self.max_workers,
            _resource_request=resource_request,
            _poll_interval=self.poll_interval,
            _coordinator=coordinator,
            _stop_event=threading.Event(),
            _use_job_arrays=self.use_job_arrays,
            _scale_thread=threading.Thread(
                target=lambda: pool_holder[0]._scale_loop(),
                name="furu-slurm-worker-pool-scale",
            ),
            _job_ids=job_ids,
            _coordinator_files=coordinator_files,
        )
        pool_holder.append(pool)
        pool._scale_thread.start()
        return pool
