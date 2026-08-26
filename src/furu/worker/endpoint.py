from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from furu.utils import atomic_replace_private_file


class WorkerEndpoint(BaseModel):
    """Who is my coordinator, and what code do I run.

    The worker's sbatch script bakes only this file's *path*; everything that
    a takeover must swap together — URL, token, project, config — lives here
    and changes under one atomic rename, so a worker can never observe them
    out of sync. Contains the auth token: never log it verbatim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    generation: int  # monotonic; a worker re-execs only if it grew
    server_url: str
    auth_token: str
    project_root: str
    config_file: str


def read_worker_endpoint(path: Path) -> WorkerEndpoint:
    return WorkerEndpoint.model_validate_json(path.read_text(encoding="utf-8"))


def write_worker_endpoint(path: Path, endpoint: WorkerEndpoint) -> None:
    atomic_replace_private_file(path, endpoint.model_dump_json(indent=2) + "\n")
