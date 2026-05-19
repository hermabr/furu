from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from furu.metadata import ArtifactSpec
from furu.resources import ResourceRequest


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lease_id: str
    artifact: ArtifactSpec


class JobCompletedResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["completed"] = "completed"


class JobFailedResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["failed"] = "failed"
    error: str


class JobBlockedResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["blocked"] = "blocked"
    dependencies: list[ArtifactSpec]


type JobResultRequest = Annotated[
    JobCompletedResult | JobFailedResult | JobBlockedResult,
    Field(discriminator="status"),
]


class OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ok: Literal[True] = True


class CountSatisfiableReadyJobsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    resource_request: ResourceRequest
    max_workers: int = Field(ge=0)


class CountSatisfiableReadyJobsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    count: int = Field(ge=0)


type LeaseJobResponse = Job | Literal["wait", "stop"]
