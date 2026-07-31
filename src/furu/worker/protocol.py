from __future__ import annotations

from typing import Annotated, Literal

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

    members: list[JobMember]
    provenance: SubmitProvenance


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


type JobResult = Annotated[
    JobCompletedResult | JobFailedResult | JobBlockedResult,
    Field(discriminator="status"),
]


class OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ok: Literal[True] = True


type LeaseJobResponse = Job | Literal["wait", "stop"]


class LeaseJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["lease_job"] = "lease_job"
    resources: ResourceRequest
    worker: str


class JobResultMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["job_result"] = "job_result"
    lease_id: str
    result: JobResult


type WorkerMessage = Annotated[
    LeaseJobRequest | JobResultMessage,
    Field(discriminator="type"),
]
