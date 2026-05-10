from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from furu.migration.types import _MigrationEdge


class _ResultLinkBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class _ResultLinkArtifact(_ResultLinkBase):
    fully_qualified_name: str
    schema_hash: str
    artifact_hash: str


class _ResultLinkSource(_ResultLinkArtifact):
    data_dir: str


class _ResultLink(_ResultLinkBase):
    kind: Literal["result_link"] = "result_link"
    current: _ResultLinkArtifact
    source: _ResultLinkSource
    migration_path: list[_MigrationEdge]
