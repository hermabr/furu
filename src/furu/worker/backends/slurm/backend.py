from __future__ import annotations

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
from furu.provenance import EnvironmentIdentity, SubmitProvenance
from furu.resources import ResourceFloor, ResourceRequest, resource_request_adapter
from furu.snapshot import extract_snapshot
from furu.utils import write_private_file
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.backends.slurm.resources import SlurmResources

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

    @property
    def resource_request(self) -> ResourceRequest:
        return ResourceRequest(
            cpus=self.resources.cpus_per_worker,
            gpus=self.resources.gpus,
            memory_gib=self.resources.memory_gib,
            reserve_for=self.reserve_for,
        )

    def start_pool(
        self,
        *,
        coordinator: ExecutionCoordinator,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        provenance: SubmitProvenance,
    ) -> SlurmWorkerPool:
        connect_port = (
            bound_port if self.worker_connect_port is None else self.worker_connect_port
        )
        server_url = f"ws://{self.worker_connect_host}:{connect_port}"

        chdir = Path.cwd().resolve()
        project_root = Path(EnvironmentIdentity.capture().project_root)
        if provenance.snapshot_id is not None:
            # Run workers from the extracted snapshot, not the live worktree,
            # so edits made after submit cannot leak into these jobs.
            # The configured snapshots directory may be relative to the submit
            # cwd.  Slurm changes into ``chdir`` before running the worker
            # script, so keep every path passed to Slurm and uv absolute.
            code_dir = extract_snapshot(provenance.snapshot_id).resolve()
            repo_root = Path(provenance.git.repo_root)
            chdir = code_dir / chdir.relative_to(repo_root)
            project_root = code_dir / Path(
                provenance.environment.project_root
            ).relative_to(repo_root)
            subprocess.run(
                ["uv", "sync", "--frozen", "--project", str(project_root)],
                env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
                check=True,
            )
        # A coordinator may run several Slurm pools, so each pool owns one
        # directory and can use fixed filenames without clobbering another.
        worker_dir = executor_dir.resolve() / "workers" / secrets.token_hex(8)
        worker_dir.mkdir(parents=True)

        token_file = worker_dir / "worker.token"
        write_private_file(token_file, auth_token, mode=0o600)

        # Workers may run from a different directory (the extracted snapshot),
        # so pin any relative data directories to the submit-side anchor.
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
                # sbatch inherits the submit environment by default.  An active
                # virtualenv belongs to the submit process, not this snapshot,
                # and makes uv warn before it selects the snapshot's .venv.
                "unset VIRTUAL_ENV\n"
                "\n"
                # --frozen forbids silent lock updates on the node; --project
                # pins the environment regardless of --chdir.
                "exec uv run --frozen "
                f"--project {shlex.quote(str(project_root))} \\\n"
                "    python -m furu.worker._cli \\\n"
                f"    --server-url {shlex.quote(server_url)} \\\n"
                f"    --auth-token-file {shlex.quote(str(token_file))} \\\n"
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
            f"--chdir={chdir}",
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
            _job_ids=[],
        )
        pool_holder.append(pool)
        pool._scale_thread.start()
        return pool
