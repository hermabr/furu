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


type JobResultRequest = Annotated[
    JobCompletedResult | JobFailedResult | JobBlockedResult,
    Field(discriminator="status"),
]


class OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["ok"] = "ok"


type LeaseJobResponse = Job | Literal["wait", "stop"]


class LeaseJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["lease_job"] = "lease_job"
    resources: ResourceRequest
    worker: str


class CountSatisfiableJobsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["count_satisfiable_jobs"] = "count_satisfiable_jobs"
    resources: ResourceRequest
    max_workers: int


class FailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["fail"] = "fail"
    message: str


class JobResultMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["job_result"] = "job_result"
    lease_id: str
    result: JobResultRequest


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["job"] = "job"
    job: Job


class WaitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["wait"] = "wait"


class StopResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["stop"] = "stop"


class CountSatisfiableJobsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["count_satisfiable_jobs"] = "count_satisfiable_jobs"
    count: int


type WorkerRequest = Annotated[
    LeaseJobRequest | JobResultMessage, Field(discriminator="type")
]
type PoolRequest = Annotated[
    CountSatisfiableJobsRequest | FailRequest, Field(discriminator="type")
]
type ClientRequest = Annotated[
    LeaseJobRequest | JobResultMessage | CountSatisfiableJobsRequest | FailRequest,
    Field(discriminator="type"),
]
type LeaseJobWireResponse = Annotated[
    JobResponse | WaitResponse | StopResponse,
    Field(discriminator="type"),
]
