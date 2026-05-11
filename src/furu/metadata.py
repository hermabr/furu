from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from furu.utils import JsonValue, object_id_from_parts

if TYPE_CHECKING:
    from furu.core import Furu


class GitData(BaseModel):
    commit: str
    branch: str
    remote: str
    # patch: dict # TODO: add this
    # submodules: dict # TODO: add this


@dataclass(frozen=True, kw_only=True)
class ArtifactSpec:
    fully_qualified_name: str
    data: dict[str, JsonValue]
    artifact_hash: str
    schema: JsonValue
    schema_hash: str

    @cached_property
    def object_id(self) -> str:
        return object_id_from_parts(
            fully_qualified_name=self.fully_qualified_name,
            schema_hash=self.schema_hash,
            artifact_hash=self.artifact_hash,
        )


class RunningMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )
    kind: Literal["running"] = "running"
    # python_def: str
    artifact: ArtifactSpec
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
    def write_for[T](
        cls,
        obj: Furu[T],
    ) -> RunningMetadata:
        metadata = cls(
            artifact=ArtifactSpec(
                fully_qualified_name=obj._fully_qualified_name,
                data=obj.artifact_data,
                artifact_hash=obj.artifact_hash,
                schema=obj.schema,
                schema_hash=obj.artifact_schema_hash,
            ),
            data_path=obj.data_dir,
            started_at=datetime.now(timezone.utc),
        )
        obj._metadata_path.write_text(metadata.model_dump_json(indent=2))
        return metadata

    def to_complete(
        self,
        *,
        observed_dependencies: tuple[str, ...],
    ) -> CompletedMetadata:
        return CompletedMetadata(
            artifact=self.artifact,
            data_path=self.data_path,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            observed_dependencies=observed_dependencies,
        )


class CompletedMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )
    kind: Literal["completed"] = "completed"
    # python_def: str
    artifact: ArtifactSpec
    data_path: Path
    # git: GitData | None
    started_at: datetime
    # traced_function_hashes: list[
    #     dict[str, str]
    # ]  # TODO: this probably means i don't really need to record the package versions? in particular since git data would have uv.lock and pyproject.toml already
    # slurm: SlurmData
    completed_at: datetime
    observed_dependencies: tuple[str, ...]


type Metadata = Annotated[
    RunningMetadata | CompletedMetadata, Field(discriminator="kind")
]
