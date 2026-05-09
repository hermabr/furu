from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from furu.serialize import from_json
from furu.utils import JsonValue

if TYPE_CHECKING:
    from furu.core import Furu


class GitData(BaseModel):
    commit: str
    branch: str
    remote: str
    # patch: dict # TODO: add this
    # submodules: dict # TODO: add this


class RunningMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        strict=True,
        revalidate_instances="always",
    )
    kind: Literal["running"] = "running"
    # python_def: str
    artifact: JsonValue
    artifact_hash: str
    schema_: JsonValue
    schema_hash: str
    data_path: Path
    # git: GitData | None
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

    @classmethod
    def write_for[T](cls, obj: Furu[T]) -> RunningMetadata:
        metadata = cls(
            artifact=obj.artifact,
            artifact_hash=obj.artifact_hash,
            schema_=obj.schema,
            schema_hash=obj.schema_hash,
            data_path=obj.data_dir,
            started_at=datetime.now(timezone.utc),
        )
        obj._metadata_path.write_text(metadata.model_dump_json(indent=2))
        return metadata

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


class CompletedMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        strict=True,
        revalidate_instances="always",
    )
    kind: Literal["completed"] = "completed"
    # python_def: str
    artifact: JsonValue
    artifact_hash: str
    schema_: JsonValue
    schema_hash: str
    data_path: Path
    # git: GitData | None
    started_at: datetime
    # traced_function_hashes: list[
    #     dict[str, str]
    # ]  # TODO: this probably means i don't really need to record the package versions? in particular since git data would have uv.lock and pyproject.toml already
    # slurm: SlurmData
    completed_at: datetime


type Metadata = Annotated[
    RunningMetadata | CompletedMetadata, Field(discriminator="kind")
]


def load_metadata(metadata: str | Path | Metadata) -> Metadata:
    if isinstance(metadata, RunningMetadata | CompletedMetadata):
        return metadata

    metadata_path = Path(metadata)
    return TypeAdapter(Metadata).validate_json(
        metadata_path.read_text(encoding="utf-8")
    )


def load_furu_from_metadata(metadata: str | Path | Metadata) -> Furu[object]:
    obj = from_json(load_metadata(metadata).artifact)
    from furu.core import Furu

    if not isinstance(obj, Furu):
        raise TypeError("Metadata artifact did not describe a Furu object")
    return obj
