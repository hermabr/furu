from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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


type FinishRequest = Annotated[
    FinishSuccessRequest | FinishFailedRequest,
    Field(discriminator="status"),
]


class BlockedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dependencies: list[ArtifactSpec]


class OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ok: Literal[True] = True


type GetJobResponse = Job | Literal["wait", "stop"]
