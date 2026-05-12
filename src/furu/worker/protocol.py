from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from furu.metadata import ArtifactSpec


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lease_id: str
    artifact: ArtifactSpec


class FinishSuccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["completed"] = "completed"


class FinishFailedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["failed"] = "failed"
    error: str


type FinishRequest = FinishSuccessRequest | FinishFailedRequest


class BlockedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dependencies: list[ArtifactSpec]


type GetJobResponse = Job | Literal["wait", "stop"]
