from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from furu.utils import (
    JsonValue,
    fully_qualified_name,
    object_id_from_parts,
    spec_label,
)

if TYPE_CHECKING:
    from furu.core import Spec


class ArtifactSpec(BaseModel):
    """A spec as a worker is sent it: its class, its fields and its id."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    fully_qualified_name: str
    # The class name in the id, which a moved class no longer has.
    stored_name: str
    artifact_data: dict[str, JsonValue]
    artifact_hash: str
    class_part: str

    @classmethod
    def from_furu[TSpec: Spec](cls, obj: TSpec) -> ArtifactSpec:
        return cls(
            fully_qualified_name=fully_qualified_name(type(obj)),
            stored_name=obj._stored_name,
            artifact_data=obj._artifact_data,
            artifact_hash=obj._artifact_hash,
            class_part=obj._class_part,
        )

    @cached_property
    def object_id(self) -> str:
        return object_id_from_parts(
            fully_qualified_name=self.stored_name,
            class_part=self.class_part,
            artifact_hash=self.artifact_hash,
        )

    @cached_property
    def log_label(self) -> str:
        return spec_label(self.stored_name, self.class_part, self.artifact_hash)
