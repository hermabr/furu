from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from furu.utils import JsonValue


class GitData(BaseModel):
    commit: str
    branch: str
    remote: str
    # patch: dict # TODO: add this
    # submodules: dict # TODO: add this


class _Metadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        strict=True,
        revalidate_instances="always",
    )
    # python_def: str
    artifact: JsonValue
    artifact_hash: str
    schema_: JsonValue
    schema_hash: str
    data_path: Path
    # git: GitData | None


class RunningMetadata(_Metadata):
    started_at: datetime
    # command: list[str] # TODO: find/decide what the most elegant approach is here
    # hostname: str
    # user: str
    # pid: int
    # TODO: what other information should i record about the person starting this run?
    # furu_package_version: str
    # TODO: include some hardware information and machine info, such as os/version
    # python_version: str
    # executor info, such as local, local executor, slurm dag or slurm worker

    def to_complete(self) -> "CompletedMetadata":
        return CompletedMetadata(
            artifact=self.artifact,
            artifact_hash=self.artifact_hash,
            schema_=self.schema_,
            schema_hash=self.schema_hash,
            data_path=self.data_path,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
        )


class CompletedMetadata(RunningMetadata):
    # traced_function_hashes: list[
    #     dict[str, str]
    # ]  # TODO: this probably means i don't really need to record the package versions? in particular since git data would have uv.lock and pyproject.toml already
    # slurm: SlurmData
    completed_at: datetime


class LockClaim(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    version: Literal[1] = 1
    lock_path: Path
    claim_path: Path
    pid: int
    hostname: str

    @model_validator(mode="after")
    def _validate_paths(self) -> "LockClaim":
        if not self.lock_path.is_absolute() or not self.claim_path.is_absolute():
            raise ValueError("lock claims require absolute paths")
        if self.claim_path.parent != self.lock_path.parent:
            raise ValueError("lock claim path must stay in the lock directory")
        if not self.claim_path.name.startswith(f"{self.lock_path.name}."):
            raise ValueError("lock claim path must use the lock basename prefix")
        if not self.claim_path.name.endswith(".claim"):
            raise ValueError("lock claim path must end with .claim")
        return self
