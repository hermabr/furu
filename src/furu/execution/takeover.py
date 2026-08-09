"""Orchestrator takeover: let a new run inherit an old run's Slurm workers.

Three cooperating pieces (see also ``furu.worker.endpoint``):

1. every coordinator registers itself in ``<executions>/live/`` while alive,
   with liveness backed by a heartbeat lock rather than trusting cleanup;
2. a successor holds a two-message handshake (inventory, then claim) on a
   single ``/takeover`` WebSocket connection to the old coordinator;
3. between the two messages the successor atomically rewrites the matched
   pools' endpoint files, so any worker that wakes up after the old server
   stops already finds the new coordinates.
"""

from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from websockets.headers import build_authorization_basic
from websockets.sync.client import connect

from furu.config import get_config
from furu.locking import is_active_lock, lock, read_text_or_none, unlink_if_exists
from furu.logging import get_logger
from furu.utils import atomic_replace_private_file
from furu.worker.endpoint import read_worker_endpoint, write_worker_endpoint

if TYPE_CHECKING:
    from furu.execution.execution_coordinator import ExecutionCoordinator
    from furu.execution.server import ExecutionCoordinatorServer
    from furu.worker.backends.protocol import WorkerBackend

logger = get_logger()

TAKEOVER_PATH = "/takeover"
_INVENTORY_TIMEOUT_S = 30.0
# The successor extracts a snapshot and builds a venv between inventory and
# claim, and the old side joins its scale threads before answering.
_CLAIM_TIMEOUT_S = 300.0
_SLURM_COMMAND_TIMEOUT_S = 60.0


class PoolJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job_id: str
    state: str


class PoolInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pool_id: str
    fingerprint: str
    endpoint_file: Path
    jobs: list[PoolJob]


class PoolsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["pools_request"] = "pools_request"
    successor_executor_id: str


class PoolsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["pools"] = "pools"
    executor_id: str
    executor_dir: Path
    pools: list[PoolInventory]


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["claim"] = "claim"
    adopt: list[str]


class ClaimResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["claimed"] = "claimed"
    adopted: list[PoolInventory]
    cancelled: list[str]


@dataclass(frozen=True, slots=True)
class AdoptedPool:
    """What a successor pool inherits: identity, indirection file, jobs."""

    pool_id: str
    endpoint_file: Path
    job_ids: tuple[str, ...]


# --------------------------------------------------------------------------
# Live-run registry


class LiveRunEntry(BaseModel):
    """Self-sufficient handle to a live coordinator: one read gives a
    successor everything. Contains the auth token: never log it verbatim."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    executor_id: str
    server_url: str
    auth_token: str
    executor_dir: Path
    pid: int
    host: str
    started_at: str


def _live_registry_dir() -> Path:
    return get_config().run_directories.executions / "live"


def _live_entry_path(executor_id: str) -> Path:
    return _live_registry_dir() / f"{executor_id}.json"


def _live_lock_path(executor_id: str) -> Path:
    return _live_registry_dir() / f"{executor_id}.lock"


@contextmanager
def register_live_run(
    *,
    executor_id: str,
    executor_dir: Path,
    bound_port: int,
    auth_token: str,
) -> Iterator[None]:
    """Advertise this coordinator while it runs.

    The heartbeat lock, not the entry file, is the liveness signal: a
    SIGKILL'd coordinator's entry goes stale with its lock and is ignored
    (and garbage-collected) by the next run that looks.
    """
    connect_host = get_config().worker.connect_host or socket.getfqdn()
    entry = LiveRunEntry(
        executor_id=executor_id,
        server_url=f"ws://{connect_host}:{bound_port}",
        auth_token=auth_token,
        executor_dir=executor_dir,
        pid=os.getpid(),
        host=socket.gethostname(),
        started_at=datetime.now(UTC).isoformat(),
    )
    _live_registry_dir().mkdir(parents=True, exist_ok=True)
    entry_path = _live_entry_path(executor_id)
    with lock(_live_lock_path(executor_id)):
        atomic_replace_private_file(entry_path, entry.model_dump_json(indent=2) + "\n")
        try:
            yield
        finally:
            unlink_if_exists(entry_path)


def _discover_live_run(selector: str, *, exclude_executor_id: str) -> LiveRunEntry:
    live_dir = _live_registry_dir()
    entries: list[LiveRunEntry] = []
    for entry_path in sorted(live_dir.glob("*.json")) if live_dir.is_dir() else []:
        raw = read_text_or_none(entry_path)
        if raw is None:
            continue
        try:
            entry = LiveRunEntry.model_validate_json(raw)
        except ValidationError:
            logger.warning("ignoring malformed live-run entry at %s", entry_path)
            continue
        if entry.executor_id == exclude_executor_id:
            continue
        if not is_active_lock(_live_lock_path(entry.executor_id)):
            logger.info("garbage-collecting stale live-run entry %s", entry.executor_id)
            unlink_if_exists(entry_path)
            continue
        entries.append(entry)
    if selector != "auto":
        entries = [entry for entry in entries if entry.executor_id.startswith(selector)]
    match entries:
        case []:
            raise RuntimeError(
                f"FURU_REPLACE_ORCHESTRATOR={selector}: no live run to replace"
            )
        case [entry]:
            return entry
        case _:
            raise RuntimeError(
                f"FURU_REPLACE_ORCHESTRATOR={selector} matches several live runs: "
                + ", ".join(entry.executor_id for entry in entries)
                + "; disambiguate with an executor-id prefix"
            )


# --------------------------------------------------------------------------
# Successor side


def perform_takeover(
    *,
    selector: str,
    coordinator: ExecutionCoordinator,
    server: ExecutionCoordinatorServer,
    worker_backends: tuple[WorkerBackend, ...],
) -> dict[int, AdoptedPool]:
    """Discover, match, rewrite, claim, signal. Returns adoptions keyed by
    the matched backend's index in ``worker_backends``."""
    from furu.worker.backends.slurm.backend import SlurmWorkerBackend

    provenance = coordinator.submit_provenance
    assert provenance is not None

    entry = _discover_live_run(selector, exclude_executor_id=coordinator.executor_id)
    logger.info("taking over coordinator %s at %s", entry.executor_id, entry.server_url)
    with connect(
        entry.server_url + TAKEOVER_PATH,
        additional_headers={
            "Authorization": build_authorization_basic("furu", entry.auth_token)
        },
        max_size=None,
    ) as connection:
        connection.send(
            PoolsRequest(
                successor_executor_id=coordinator.executor_id
            ).model_dump_json()
        )
        inventory = PoolsResponse.model_validate_json(
            connection.recv(timeout=_INVENTORY_TIMEOUT_S)
        )

        unmatched = {pool.pool_id: pool for pool in inventory.pools}
        matched: list[tuple[int, SlurmWorkerBackend, PoolInventory]] = []
        for index, backend in enumerate(worker_backends):
            if not isinstance(backend, SlurmWorkerBackend):
                continue
            fingerprint = backend.fingerprint()
            pool = next(
                (p for p in unmatched.values() if p.fingerprint == fingerprint), None
            )
            if pool is None:
                continue
            del unmatched[pool.pool_id]
            matched.append((index, backend, pool))
        if not matched:
            raise RuntimeError(
                "FURU_REPLACE_ORCHESTRATOR: no pool of the live run "
                f"{inventory.executor_id} matches this run's backends; "
                "old pool fingerprints: "
                + (
                    ", ".join(
                        f"{pool.pool_id}={pool.fingerprint}" for pool in inventory.pools
                    )
                    or "<none>"
                )
                + "; new backend fingerprints: "
                + (
                    ", ".join(
                        backend.fingerprint()
                        for backend in worker_backends
                        if isinstance(backend, SlurmWorkerBackend)
                    )
                    or "<none>"
                )
                + ". Unset FURU_REPLACE_ORCHESTRATOR to start cold."
            )
        if unmatched:
            logger.info(
                "no backend matches old pools %s; the old run will cancel them",
                ", ".join(sorted(unmatched)),
            )

        # Rewrite endpoint files before claiming: from the moment the old
        # server can shut down, any worker that wakes up must already find
        # the new coordinates.
        for _, backend, pool in matched:
            old_endpoint = read_worker_endpoint(pool.endpoint_file)
            write_worker_endpoint(
                pool.endpoint_file,
                backend.takeover_endpoint(
                    generation=old_endpoint.generation + 1,
                    bound_port=server.bound_port,
                    auth_token=server.auth_token,
                    executor_dir=coordinator.executor_dir,
                    provenance=provenance,
                ),
            )

        connection.send(
            ClaimRequest(
                adopt=[pool.pool_id for _, _, pool in matched]
            ).model_dump_json()
        )
        response = ClaimResponse.model_validate_json(
            connection.recv(timeout=_CLAIM_TIMEOUT_S)
        )

    claimed = {pool.pool_id: pool for pool in response.adopted}
    adoptions: dict[int, AdoptedPool] = {}
    signal_job_ids: list[str] = []
    for index, _, pool in matched:
        final = claimed.get(pool.pool_id, pool)
        adoptions[index] = AdoptedPool(
            pool_id=pool.pool_id,
            endpoint_file=pool.endpoint_file,
            job_ids=tuple(job.job_id for job in final.jobs),
        )
        signal_job_ids.extend(
            job.job_id for job in final.jobs if job.state == "RUNNING"
        )
        logger.info(
            "adopted pool %s · %d jobs · old worker logs remain under %s",
            pool.pool_id,
            len(final.jobs),
            inventory.executor_dir,
        )
    if response.cancelled:
        logger.info(
            "old run cancels its unadopted pools: %s", ", ".join(response.cancelled)
        )
    _signal_adopted_workers(signal_job_ids)
    return adoptions


def _signal_adopted_workers(job_ids: list[str]) -> None:
    """Interrupt busy workers without killing their allocations; each unwinds,
    re-reads its endpoint file, and re-execs into the new snapshot. On failure
    a busy worker still redirects at its next touch of the dead old socket."""
    if not job_ids:
        return
    try:
        result = subprocess.run(
            ["scancel", "--signal=USR1", "--batch", *job_ids],
            check=False,
            capture_output=True,
            text=True,
            timeout=_SLURM_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        logger.warning("scancel --signal=USR1 timed out for %s", ",".join(job_ids))
        return
    if result.returncode != 0:
        logger.warning(
            "scancel --signal=USR1 failed for %s: %s",
            ",".join(job_ids),
            result.stderr.strip(),
        )
