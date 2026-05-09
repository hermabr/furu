from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from furu.utils import JsonValue

if TYPE_CHECKING:
    from furu.core import Furu


class GitData(BaseModel):
    commit: str
    branch: str
    remote: str
    # patch: dict # TODO: add this
    # submodules: dict # TODO: add this


@dataclass(frozen=True, kw_only=True)
class ArtifactMetadata:
    data: JsonValue
    hash: str
    schema: JsonValue
    schema_hash: str


class DependencyRef(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    object_id: str
    class_name: str
    data_path: Path
    artifact_hash: str
    artifact_schema_hash: str
    via: Literal["field", "dependency", "load_or_create", "try_load"]
    path: str | None = None

    @classmethod
    def from_furu(
        cls,
        obj: Furu[Any],
        *,
        via: Literal["field", "dependency", "load_or_create", "try_load"],
        path: str | None = None,
    ) -> DependencyRef:
        return cls(
            object_id=obj.object_id,
            class_name=obj._fully_qualified_name,
            data_path=obj.data_dir,
            artifact_hash=obj.artifact_hash,
            artifact_schema_hash=obj.artifact_schema_hash,
            via=via,
            path=path,
        )


class DependencyMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    eager: tuple[DependencyRef, ...] = ()
    lazy: tuple[DependencyRef, ...] = ()


class RunningMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )
    kind: Literal["running"] = "running"
    # python_def: str
    artifact: ArtifactMetadata
    data_path: Path
    # git: GitData | None
    started_at: datetime
    dependencies: DependencyMetadata = DependencyMetadata()
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
        *,
        dependencies: DependencyMetadata | None = None,
    ) -> RunningMetadata:
        metadata = cls(
            artifact=ArtifactMetadata(
                data=obj.artifact_data,
                hash=obj.artifact_hash,
                schema=obj.schema,
                schema_hash=obj.artifact_schema_hash,
            ),
            data_path=obj.data_dir,
            started_at=datetime.now(timezone.utc),
            dependencies=dependencies or DependencyMetadata(),
        )
        obj._metadata_path.write_text(metadata.model_dump_json(indent=2))
        return metadata

    def to_complete(
        self, *, dependencies: DependencyMetadata | None = None
    ) -> CompletedMetadata:
        return CompletedMetadata(
            artifact=self.artifact,
            data_path=self.data_path,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            dependencies=dependencies or self.dependencies,
        )


class CompletedMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )
    kind: Literal["completed"] = "completed"
    # python_def: str
    artifact: ArtifactMetadata
    data_path: Path
    # git: GitData | None
    started_at: datetime
    # traced_function_hashes: list[
    #     dict[str, str]
    # ]  # TODO: this probably means i don't really need to record the package versions? in particular since git data would have uv.lock and pyproject.toml already
    # slurm: SlurmData
    completed_at: datetime
    dependencies: DependencyMetadata = DependencyMetadata()


type Metadata = Annotated[
    RunningMetadata | CompletedMetadata, Field(discriminator="kind")
]
