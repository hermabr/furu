from __future__ import annotations

from typing import TYPE_CHECKING, Any

from furu.metadata import ArtifactSpec

if TYPE_CHECKING:
    from furu.core import Furu


def artifact_spec_for(obj: Furu[Any]) -> ArtifactSpec:
    return ArtifactSpec(
        fully_qualified_name=obj._fully_qualified_name,
        data=obj.artifact_data,
        artifact_hash=obj.artifact_hash,
        schema=obj.schema,
        schema_hash=obj.artifact_schema_hash,
    )
