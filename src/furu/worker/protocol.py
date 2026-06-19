from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from furu.metadata import ArtifactSpec, FuruSpec
from furu.resources import ResourceRequest
from furu.utils import JsonValue


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lease_id: str
    artifact: ArtifactSpec
    runtime_data: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def from_furu(cls, *, lease_id: str, obj: object) -> Job:
        from furu.core import Furu

        if not isinstance(obj, Furu):
            raise TypeError("Job.from_furu expects a Furu object")
        return cls(
            lease_id=lease_id,
            artifact=ArtifactSpec.from_furu(obj),
            runtime_data=obj._runtime_data,
        )


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
    dependencies: list[FuruSpec]


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


class CountSatisfiableJobsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resources: ResourceRequest
    max_workers: int


class FailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str
