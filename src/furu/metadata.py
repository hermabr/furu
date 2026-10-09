from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from furu.utils import (
    JsonValue,
    object_id_from_parts,
    spec_label,
)

if TYPE_CHECKING:
    from furu.core import Spec


class ArtifactSpec(BaseModel):
    """Which spec an identity directory holds; stored there as spec.json."""

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
    def from_furu[TSpec: Spec](cls, obj: TSpec) -> ArtifactSpec:
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

    @cached_property
    def log_label(self) -> str:
        return spec_label(
            self.fully_qualified_name, self.schema_hash, self.artifact_hash
        )
