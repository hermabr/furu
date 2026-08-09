from __future__ import annotations

import hashlib
import json
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
from furu.execution.takeover import AdoptedPool, PoolOffer
from furu.provenance import EnvironmentIdentity, SubmitProvenance
from furu.resources import ResourceRequest
from furu.snapshot import extract_snapshot
from furu.utils import write_private_file
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.backends.slurm.resources import SlurmResources
from furu.worker.endpoint import WorkerEndpoint, write_worker_endpoint

if TYPE_CHECKING:
    from furu.execution.execution_coordinator import ExecutionCoordinator

type SlurmExport = Literal["NIL", "ALL"] | tuple[str, ...] | None


def _endpoint_field_lookup(field_name: str) -> str:
    """Shell fragment resolving one endpoint-file field at script runtime."""
    return (
        '"$(python3 -c '
        "'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "
        f'"$furu_endpoint_file" {field_name})"'
    )


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

    def fingerprint(self) -> str:
        """Hash of exactly the fields that change what a worker *is*.

        Submitter-side concerns (max_workers, poll interval, idle timeout,
        connect host/port) and the snapshot are deliberately excluded: an old
        pool whose fingerprint matches can serve this backend's jobs.
        """
        payload = json.dumps(
            {
                "sbatch_args": self.resources.to_sbatch_args(),
                "job_name": self.job_name,
                "export": self.export,
                "use_job_arrays": self.use_job_arrays,
                "pre_worker_commands": self.pre_worker_commands,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _server_url(self, bound_port: int) -> str:
        connect_port = (
            bound_port if self.worker_connect_port is None else self.worker_connect_port
        )
        return f"ws://{self.worker_connect_host}:{connect_port}"

    def _prepare_project(self, provenance: SubmitProvenance) -> tuple[Path, Path]:
        """Resolve (chdir, project_root), extracting the snapshot and building
        its venv so workers never race to create it."""
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
        return chdir, project_root

    def _write_worker_config(self, worker_dir: Path) -> Path:
        # Workers may run from a different directory (the extracted snapshot),
        # so pin any relative data directories to the submit-side anchor.
        config = get_config()
        config = config.model_copy(
            update={"directories": config.directories.anchored()}
        )
        config_file = worker_dir / f"worker-{secrets.token_hex(16)}.config.json"
        write_private_file(
            config_file,
            config.model_dump_json(indent=2) + "\n",
            mode=0o600,
        )
        return config_file

    def takeover_offer(
        self,
        *,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        provenance: SubmitProvenance,
    ) -> PoolOffer:
        """This run's coordinates for adopted workers, with the snapshot venv
        built and this run's worker config written."""
        _, project_root = self._prepare_project(provenance)
        worker_dir = executor_dir.resolve() / "workers"
        worker_dir.mkdir(parents=True, exist_ok=True)
        return PoolOffer(
            fingerprint=self.fingerprint(),
            server_url=self._server_url(bound_port),
            auth_token=auth_token,
            project_root=str(project_root),
            config_file=str(self._write_worker_config(worker_dir)),
        )

    def start_pool(
        self,
        *,
        coordinator: ExecutionCoordinator,
        bound_port: int,
        auth_token: str,
        executor_dir: Path,
        provenance: SubmitProvenance,
        adopt: AdoptedPool | None = None,
    ) -> SlurmWorkerPool:
        chdir, project_root = self._prepare_project(provenance)
        worker_dir = executor_dir.resolve() / "workers"
        worker_dir.mkdir(parents=True, exist_ok=True)

        if adopt is None:
            pool_id = secrets.token_hex(8)
            endpoint_file = worker_dir / f"endpoint-{pool_id}.json"
            write_worker_endpoint(
                endpoint_file,
                WorkerEndpoint(
                    generation=1,
                    server_url=self._server_url(bound_port),
                    auth_token=auth_token,
                    project_root=str(project_root),
                    config_file=str(self._write_worker_config(worker_dir)),
                ),
            )
        else:
            # The takeover already rewrote the adopted pool's endpoint file to
            # point here; the pool's queued job scripts bake that path, so
            # this run's new submissions must keep using the same file.
            pool_id = adopt.pool_id
            endpoint_file = adopt.endpoint_file

        resource_request = ResourceRequest(
            cpus=self.resources.cpus_per_worker,
            gpus=self.resources.gpus,
            memory_gib=self.resources.memory_gib,
        )
        pre_worker_script = "".join(
            f"{command}\n" for command in self.pre_worker_commands
        )
        if pre_worker_script:
            pre_worker_script += "\n"

        scripts_dir = worker_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"worker-{pool_id}.sh"
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
                # The endpoint file is the one runtime indirection between this
                # (copied-at-submit) script and its coordinator: resolving the
                # config, project, and server here rather than at submit time
                # is what lets a successor run redirect already-queued jobs.
                f"furu_endpoint_file={shlex.quote(str(endpoint_file))}\n"
                "\n"
                f"{component_line}"
                "\n"
                f"{pre_worker_script}"
                # sbatch inherits the submit environment by default.  An active
                # virtualenv belongs to the submit process, not this snapshot,
                # and makes uv warn before it selects the snapshot's .venv.
                "unset VIRTUAL_ENV\n"
                "\n"
                "export "
                f"{_WORKER_JSON_CONFIG_FILE_ENV_VAR}="
                f"{_endpoint_field_lookup('config_file')}\n"
                "\n"
                # --frozen forbids silent lock updates on the node; --project
                # pins the environment regardless of --chdir.
                "exec uv run --frozen "
                f"--project {_endpoint_field_lookup('project_root')} \\\n"
                "    python -m furu.worker._cli \\\n"
                '    --endpoint-file "$furu_endpoint_file" \\\n'
                '    --component "${furu_worker_component}" \\\n'
                "    --backend slurm \\\n"
                f"    --idle-timeout {self.worker_idle_timeout} \\\n"
                f"    --resource-cpus {resource_request.cpus} \\\n"
                f"    --resource-gpus {resource_request.gpus} \\\n"
                f"    --resource-memory-gib {resource_request.memory_gib}\n"
            ),
            mode=0o700,
        )

        log_dir = worker_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
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
            _job_ids=list(adopt.job_ids) if adopt is not None else [],
            _pool_id=pool_id,
            _fingerprint=self.fingerprint(),
            _endpoint_file=endpoint_file,
            _surrendered=threading.Event(),
        )
        pool_holder.append(pool)
        pool._scale_thread.start()
        return pool
