from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from test_slurm_backend import (
    _disable_slurm_pool_scale_thread,
    _install_fake_slurm,
    _read_records,
    _submit_provenance,
)
from websockets.exceptions import ConnectionClosed
from websockets.headers import build_authorization_basic
from websockets.sync.client import ClientConnection, connect

import furu
from furu.config import _Config, get_config
from furu.dag import _add_to_dag
from furu.execution.execution_coordinator import (
    ExecutionCoordinator,
    FuruReplacedError,
)
from furu.execution.server import (
    ExecutionCoordinatorServer,
    execution_coordinator_server,
)
from furu.execution.takeover import (
    TAKEOVER_PATH,
    AdoptedPool,
    LiveRunEntry,
    PoolOffer,
    TakeoverRequest,
    TakeoverResponse,
    _discover_live_run,
    _live_registry_dir,
    perform_takeover,
    register_live_run,
)
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.pool import SlurmWorkerPool
from furu.worker.backends.slurm.resources import SlurmResources
from furu.worker.endpoint import (
    WorkerEndpoint,
    read_worker_endpoint,
    write_worker_endpoint,
)

OLD_EXECUTOR_ID = "0123456789abcdef" * 2
NEW_EXECUTOR_ID = "fedcba9876543210" * 2


def _coordinator(executor_id: str) -> ExecutionCoordinator:
    coordinator = ExecutionCoordinator(max_retries_per_object=0)
    coordinator.executor_id = executor_id
    return coordinator


@contextmanager
def _loopback_connect_host() -> Iterator[None]:
    """The live registry advertises the configured connect host; point it at
    the loopback interface these test servers actually bind."""
    data = get_config().model_dump()
    data["worker"]["connect_host"] = "127.0.0.1"
    with override_config(_Config.model_validate(data)):
        yield


# ---------------------------------------------------------------------------
# Live-run registry


def test_register_live_run_is_discoverable_while_held() -> None:
    with register_live_run(
        executor_id=OLD_EXECUTOR_ID,
        executor_dir=Path("/somewhere/executions") / OLD_EXECUTOR_ID,
        bound_port=4567,
        auth_token="registry-token",
    ):
        entry = _discover_live_run("auto", exclude_executor_id=NEW_EXECUTOR_ID)
        assert entry.executor_id == OLD_EXECUTOR_ID
        assert entry.auth_token == "registry-token"
        assert entry.server_url.endswith(":4567")
        entry_path = _live_registry_dir() / f"{OLD_EXECUTOR_ID}.json"
        assert stat.S_IMODE(entry_path.stat().st_mode) == 0o600

        # A run never discovers itself.
        with pytest.raises(RuntimeError, match="no live run"):
            _discover_live_run("auto", exclude_executor_id=OLD_EXECUTOR_ID)

    with pytest.raises(RuntimeError, match="no live run"):
        _discover_live_run("auto", exclude_executor_id=NEW_EXECUTOR_ID)


def test_discover_garbage_collects_entries_without_active_heartbeat() -> None:
    live_dir = _live_registry_dir()
    live_dir.mkdir(parents=True, exist_ok=True)
    stale_path = live_dir / f"{OLD_EXECUTOR_ID}.json"
    stale_path.write_text(
        LiveRunEntry(
            executor_id=OLD_EXECUTOR_ID,
            server_url="ws://gone:1",
            auth_token="stale",
            executor_dir=Path("/gone"),
            pid=1,
            host="gone",
            started_at="2026-01-01T00:00:00+00:00",
        ).model_dump_json()
    )

    with pytest.raises(RuntimeError, match="no live run"):
        _discover_live_run("auto", exclude_executor_id=NEW_EXECUTOR_ID)
    assert not stale_path.exists()


def test_discover_requires_disambiguation_and_accepts_prefix() -> None:
    other_id = "aaaa" + OLD_EXECUTOR_ID[4:]
    with (
        register_live_run(
            executor_id=OLD_EXECUTOR_ID,
            executor_dir=Path("/a"),
            bound_port=1,
            auth_token="a",
        ),
        register_live_run(
            executor_id=other_id,
            executor_dir=Path("/b"),
            bound_port=2,
            auth_token="b",
        ),
    ):
        with pytest.raises(RuntimeError, match="matches several live runs"):
            _discover_live_run("auto", exclude_executor_id=NEW_EXECUTOR_ID)
        entry = _discover_live_run("aaaa", exclude_executor_id=NEW_EXECUTOR_ID)
        assert entry.executor_id == other_id


# ---------------------------------------------------------------------------
# Takeover request handling (old-coordinator side)


@contextmanager
def _old_run_with_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_lines: str = "100_0 RUNNING\n100_1 PENDING\n",
) -> Iterator[
    tuple[ExecutionCoordinator, ExecutionCoordinatorServer, SlurmWorkerPool, Path]
]:
    _disable_slurm_pool_scale_thread(monkeypatch)
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    coordinator = _coordinator(OLD_EXECUTOR_ID)
    backend = SlurmWorkerBackend(
        max_workers=2,
        resources=SlurmResources(cpus_per_worker=1),
        worker_connect_host="127.0.0.1",
    )
    with (
        _loopback_connect_host(),
        execution_coordinator_server(
            coordinator, bind_host="127.0.0.1", port=0
        ) as server,
    ):
        pool = backend.start_pool(
            coordinator=coordinator,
            bound_port=server.bound_port,
            auth_token=server.auth_token,
            executor_dir=coordinator.executor_dir,
            provenance=_submit_provenance(),
        )
        coordinator.pools.append(pool)
        pool._job_ids[:] = ["100_0", "100_1"]
        active_file.write_text(job_lines)
        yield coordinator, server, pool, record_file


def _takeover_connection(server: ExecutionCoordinatorServer) -> ClientConnection:
    return connect(
        f"ws://127.0.0.1:{server.bound_port}{TAKEOVER_PATH}",
        additional_headers={
            "Authorization": build_authorization_basic("furu", server.auth_token)
        },
    )


def _offer(fingerprint: str) -> PoolOffer:
    return PoolOffer(
        fingerprint=fingerprint,
        server_url="ws://successor.test:9",
        auth_token="successor-token",
        project_root="/proj/new",
        config_file="/cfg/new.json",
    )


def _scancel_records(record_file: Path) -> list[list[str]]:
    return [
        record["argv"]
        for record in _read_records(record_file)
        if record["executable"] == "scancel"
    ]


def test_takeover_request_surrenders_matching_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _old_run_with_pool(tmp_path, monkeypatch) as (
        coordinator,
        server,
        pool,
        record_file,
    ):
        with _takeover_connection(server) as connection:
            connection.send(
                TakeoverRequest(
                    successor_executor_id=NEW_EXECUTOR_ID,
                    offers=[_offer(pool.fingerprint)],
                ).model_dump_json()
            )
            response = TakeoverResponse.model_validate_json(connection.recv(timeout=5))
            assert response.executor_id == OLD_EXECUTOR_ID
            assert response.pool_fingerprints == [pool.fingerprint]
            assert response.cancelled == []
            assert response.adopted == {
                0: AdoptedPool(
                    pool_id=pool.pool_id,
                    endpoint_file=pool._endpoint_file,
                    job_ids=("100_0", "100_1"),
                )
            }

        # The endpoint file now points at the successor, atomically.
        endpoint = read_worker_endpoint(pool._endpoint_file)
        assert endpoint.generation == 2
        assert endpoint.server_url == "ws://successor.test:9"
        assert endpoint.auth_token == "successor-token"
        assert endpoint.project_root == "/proj/new"
        assert endpoint.config_file == "/cfg/new.json"

        # Only the RUNNING job is signalled; PENDING jobs need nothing.
        assert _scancel_records(record_file) == [["--signal=USR1", "--batch", "100_0"]]

        assert coordinator.done.wait(timeout=5)
        with pytest.raises(FuruReplacedError, match=NEW_EXECUTOR_ID):
            coordinator.raise_for_failure()

        # The ordinary shutdown path must leave the surrendered jobs alone.
        pool.stop(timeout=0)
        assert _scancel_records(record_file) == [["--signal=USR1", "--batch", "100_0"]]


def test_takeover_request_without_matching_offer_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _old_run_with_pool(tmp_path, monkeypatch) as (
        coordinator,
        server,
        pool,
        record_file,
    ):
        with _takeover_connection(server) as connection:
            connection.send(
                TakeoverRequest(
                    successor_executor_id=NEW_EXECUTOR_ID,
                    offers=[_offer("sha256:" + "0" * 64)],
                ).model_dump_json()
            )
            response = TakeoverResponse.model_validate_json(connection.recv(timeout=5))
            assert response.adopted == {}
            assert response.cancelled == []
            assert response.pool_fingerprints == [pool.fingerprint]

        assert not coordinator.done.wait(timeout=0.2)
        assert not pool._surrendered.is_set()
        assert coordinator.replaced_by is None
        assert read_worker_endpoint(pool._endpoint_file).generation == 1
        assert _scancel_records(record_file) == []


@pytest.mark.parametrize("payload", [None, "not a takeover request"])
def test_successor_drop_or_malformed_request_keeps_all_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str | None,
) -> None:
    with _old_run_with_pool(tmp_path, monkeypatch) as (coordinator, server, pool, _):
        with _takeover_connection(server) as connection:
            if payload is not None:
                connection.send(payload)
                with pytest.raises(ConnectionClosed):
                    connection.recv(timeout=5)

        assert not coordinator.done.wait(timeout=0.5)
        assert not pool._surrendered.is_set()
        assert coordinator.replaced_by is None
        assert read_worker_endpoint(pool._endpoint_file).generation == 1


def test_second_takeover_connection_is_rejected_as_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        _old_run_with_pool(tmp_path, monkeypatch) as (_, server, _, _),
        _takeover_connection(server) as _first,
    ):
        # The handler claims the busy slot on connect, before the request;
        # give its thread a moment to get there.
        time.sleep(0.2)

        with _takeover_connection(server) as second:
            with pytest.raises(ConnectionClosed) as exc_info:
                second.recv(timeout=5)
            assert exc_info.value.rcvd is not None
            assert exc_info.value.rcvd.code == 1013


# ---------------------------------------------------------------------------
# perform_takeover (successor side)


def test_perform_takeover_adopts_matching_pool_and_signals_running_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _old_run_with_pool(tmp_path, monkeypatch) as (
        old_coordinator,
        _old_server,
        old_pool,
        record_file,
    ):
        old_endpoint_file = old_pool._endpoint_file
        assert read_worker_endpoint(old_endpoint_file).generation == 1

        new_coordinator = _coordinator(NEW_EXECUTOR_ID)
        new_coordinator.submit_provenance = _submit_provenance()
        matching_backend = SlurmWorkerBackend(
            max_workers=5,  # deliberately different: excluded from the fingerprint
            resources=SlurmResources(cpus_per_worker=1),
            worker_connect_host="127.0.0.1",
        )
        unmatched_backend = SlurmWorkerBackend(
            max_workers=1,
            resources=SlurmResources(cpus_per_worker=32),
            worker_connect_host="127.0.0.1",
        )

        with execution_coordinator_server(
            new_coordinator, bind_host="127.0.0.1", port=0
        ) as new_server:
            adoptions = perform_takeover(
                selector="auto",
                coordinator=new_coordinator,
                server=new_server,
                worker_backends=(unmatched_backend, matching_backend),
            )

            assert adoptions == {
                1: AdoptedPool(
                    pool_id=old_pool.pool_id,
                    endpoint_file=old_endpoint_file,
                    job_ids=("100_0", "100_1"),
                )
            }

            # The endpoint file now points at the successor, atomically.
            endpoint = read_worker_endpoint(old_endpoint_file)
            assert endpoint.generation == 2
            assert endpoint.server_url == f"ws://127.0.0.1:{new_server.bound_port}"
            assert endpoint.auth_token == new_server.auth_token
            assert Path(endpoint.config_file).is_relative_to(
                new_coordinator.executor_dir
            )

            # Only the RUNNING job is signalled; PENDING jobs need nothing.
            assert _scancel_records(record_file) == [
                ["--signal=USR1", "--batch", "100_0"]
            ]

            assert old_coordinator.done.wait(timeout=5)
            with pytest.raises(FuruReplacedError, match=NEW_EXECUTOR_ID):
                old_coordinator.raise_for_failure()

            # The successor's pool starts seeded with the inherited jobs and
            # keeps submitting new workers against the same endpoint file.
            new_pool = matching_backend.start_pool(
                coordinator=new_coordinator,
                bound_port=new_server.bound_port,
                auth_token=new_server.auth_token,
                executor_dir=new_coordinator.executor_dir,
                provenance=new_coordinator.submit_provenance,
                adopt=adoptions[1],
            )
            assert new_pool._job_ids == ["100_0", "100_1"]
            assert new_pool.pool_id == old_pool.pool_id
            assert new_pool._endpoint_file == old_endpoint_file


def test_perform_takeover_errors_when_no_pool_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _old_run_with_pool(tmp_path, monkeypatch) as (old_coordinator, _, pool, _):
        new_coordinator = _coordinator(NEW_EXECUTOR_ID)
        new_coordinator.submit_provenance = _submit_provenance()
        mismatched_backend = SlurmWorkerBackend(
            max_workers=1,
            resources=SlurmResources(cpus_per_worker=32),
            worker_connect_host="127.0.0.1",
        )

        with (
            execution_coordinator_server(
                new_coordinator, bind_host="127.0.0.1", port=0
            ) as new_server,
            pytest.raises(RuntimeError, match="no pool of the live run"),
        ):
            perform_takeover(
                selector="auto",
                coordinator=new_coordinator,
                server=new_server,
                worker_backends=(mismatched_backend,),
            )

        # The old run is unaffected: nothing was adopted.
        assert not old_coordinator.done.wait(timeout=0.5)
        assert not pool._surrendered.is_set()
        assert read_worker_endpoint(pool._endpoint_file).generation == 1


def test_perform_takeover_requires_a_slurm_backend() -> None:
    new_coordinator = _coordinator(NEW_EXECUTOR_ID)
    new_coordinator.submit_provenance = _submit_provenance()

    with (
        register_live_run(
            executor_id=OLD_EXECUTOR_ID,
            executor_dir=Path("/a"),
            bound_port=1,
            auth_token="a",
        ),
        pytest.raises(RuntimeError, match="no Slurm worker backends"),
    ):
        perform_takeover(
            selector="auto",
            coordinator=new_coordinator,
            server=ExecutionCoordinatorServer(
                bound_host="127.0.0.1", bound_port=1, auth_token="x"
            ),
            worker_backends=(LocalThreadWorkerBackend(),),
        )


# ---------------------------------------------------------------------------
# ExecutionCoordinator.run wiring


class _TakeoverProbeSpec(furu.Spec):
    value: int = 0

    def create(self) -> int:
        return self.value


def test_run_errors_when_replace_is_requested_but_nothing_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURU_REPLACE_ORCHESTRATOR", "auto")

    with pytest.raises(RuntimeError, match="no live run to replace"):
        ExecutionCoordinator.run(
            [_TakeoverProbeSpec(value=1)],
            worker_backends=(LocalThreadWorkerBackend(),),
        )


def test_run_treats_empty_replace_selector_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURU_REPLACE_ORCHESTRATOR", "")

    (obj,) = ExecutionCoordinator.run(
        [_TakeoverProbeSpec(value=2)],
        worker_backends=(LocalThreadWorkerBackend(),),
    )
    assert furu.create(obj) == 2


# ---------------------------------------------------------------------------
# End to end: a real worker process redirects to a successor coordinator


@pytest.mark.skipif(shutil.which("uv") is None, reason="requires uv")
def test_worker_process_redirects_to_successor_and_completes_work(
    tmp_path: Path,
) -> None:
    """An idle worker of the old run must, once the endpoint file is rewritten
    and the old server goes away, re-exec and complete the new run's work."""
    config = get_config()
    config = config.model_copy(update={"directories": config.directories.anchored()})
    config_file = tmp_path / "worker.config.json"
    config_file.write_text(config.model_dump_json(indent=2))
    project_root = EnvironmentIdentity.capture().project_root

    provenance = SubmitProvenance(
        git=GitIdentity(
            commit="0" * 40,
            branch=None,
            remote=None,
            repo_root=".",
            dirty=False,
            diff_stats=None,
        ),
        # Real environment identity so the worker-side lock-hash check passes.
        environment=EnvironmentIdentity.capture(),
        snapshot_id=None,
        submitted=SubmitContext.capture(),
    )

    old_coordinator = _coordinator(OLD_EXECUTOR_ID)
    obj = _TakeoverProbeSpec(value=7)
    new_coordinator = _coordinator(NEW_EXECUTOR_ID)
    new_coordinator.submit_provenance = provenance
    _add_to_dag(new_coordinator, [obj])

    worker_env = dict(os.environ)
    worker_env["_FURU_WORKER_JSON_CONFIG_FILE"] = str(config_file)
    # The worker resolves this test module's Spec class by qualified name.
    worker_env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), worker_env.get("PYTHONPATH", "")]
    )
    worker_env.pop("VIRTUAL_ENV", None)

    endpoint_file = tmp_path / "endpoint.json"
    worker: subprocess.Popen[bytes] | None = None
    try:
        with execution_coordinator_server(
            new_coordinator, bind_host="127.0.0.1", port=0
        ) as new_server:
            with execution_coordinator_server(
                old_coordinator, bind_host="127.0.0.1", port=0
            ) as old_server:
                write_worker_endpoint(
                    endpoint_file,
                    WorkerEndpoint(
                        generation=1,
                        server_url=f"ws://127.0.0.1:{old_server.bound_port}",
                        auth_token=old_server.auth_token,
                        project_root=project_root,
                        config_file=str(config_file),
                    ),
                )
                worker = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "furu.worker._cli",
                        "--endpoint-file",
                        str(endpoint_file),
                        "--resource-cpus",
                        "1",
                        "--resource-gpus",
                        "0",
                        "--resource-memory-gib",
                        "0",
                        "--idle-timeout",
                        "120",
                        "--component",
                        "e2e-worker",
                        "--backend",
                        "slurm",
                    ],
                    env=worker_env,
                )
                # Give the worker a moment to read generation 1 and connect;
                # if it is slower than this it starts against the rewritten
                # endpoint instead — the (equally valid) queued-job path.
                time.sleep(2.0)
                write_worker_endpoint(
                    endpoint_file,
                    WorkerEndpoint(
                        generation=2,
                        server_url=f"ws://127.0.0.1:{new_server.bound_port}",
                        auth_token=new_server.auth_token,
                        project_root=project_root,
                        config_file=str(config_file),
                    ),
                )
            # The old server is gone; the worker re-reads the endpoint file,
            # re-execs, connects to the successor, and executes its job.
            assert new_coordinator.done.wait(timeout=120)
        new_coordinator.raise_for_failure()
        assert worker.wait(timeout=60) == 0
    finally:
        if worker is not None and worker.poll() is None:
            worker.kill()

    assert furu.load_existing([obj]) == [7]
