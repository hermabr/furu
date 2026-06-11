from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from furu._storage_layout import metadata_path_in
from furu.utils import JsonValue, object_id_from_parts

if TYPE_CHECKING:
    from furu.core import Furu


class ArtifactSpec(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    fully_qualified_name: str
    artifact_data: dict[str, JsonValue]
    artifact_hash: str
    schema_data: JsonValue
    schema_hash: str

    @classmethod
    def from_furu[TFuru: Furu](cls, obj: TFuru) -> ArtifactSpec:
        return cls(
            fully_qualified_name=obj._fully_qualified_name,
            artifact_data=obj._artifact_data,
            artifact_hash=obj._artifact_hash,
            schema_data=obj._schema_data,
            schema_hash=obj._artifact_schema_hash,
        )

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
    artifact: ArtifactSpec
    base_path: Path
    started_at: datetime

    @classmethod
    def write_for[T](
        cls,
        obj: Furu[T],
    ) -> RunningMetadata:
        metadata = cls(
            artifact=ArtifactSpec.from_furu(obj),
            base_path=obj._base_dir,
            started_at=datetime.now(timezone.utc),
        )
        metadata_path_in(obj._base_dir).write_text(metadata.model_dump_json(indent=2))
        return metadata

    def to_complete(
        self,
        *,
        observed_dependencies: tuple[str, ...],
    ) -> CompletedMetadata:
        return CompletedMetadata(
            artifact=self.artifact,
            base_path=self.base_path,
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
    artifact: ArtifactSpec
    base_path: Path
    started_at: datetime
    completed_at: datetime
    observed_dependencies: tuple[str, ...]
