from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from furu.metadata import ArtifactSpec
from furu.provenance import SubmitProvenance
from furu.resources import ResourceRequest


class JobMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lease_id: str
    artifact: ArtifactSpec


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    members: list[JobMember] = Field(min_length=1)
    provenance: SubmitProvenance

    @property
    def lease_id(self) -> str:
        return self.members[0].lease_id

    @property
    def artifact(self) -> ArtifactSpec:
        return self.members[0].artifact


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


JobResultRequest: TypeAlias = Annotated[
    JobCompletedResult | JobFailedResult | JobBlockedResult,
    Field(discriminator="status"),
]


class OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ok: Literal[True] = True


LeaseJobResponse: TypeAlias = Job | Literal["wait", "stop"]


class LeaseJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resources: ResourceRequest
    worker: str


class CountSatisfiableJobsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resources: ResourceRequest
    max_workers: int


class FailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str


class WorkerLostRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker: str
