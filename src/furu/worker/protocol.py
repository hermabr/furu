from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from furu.metadata import ArtifactSpec
from furu.provenance import SubmitProvenance
from furu.resources import ResourceRequest
from furu.spec_metadata import Metadata, Reuse


class ProcessSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    environment: dict[str, str | None]
    required_environment_variables: tuple[str, ...]
    reuse: Reuse

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> ProcessSettings:
        return cls(
            environment=dict(metadata.environment),
            required_environment_variables=metadata.required_environment_variables,
            reuse=metadata.reuse,
        )


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifacts: list[ArtifactSpec]
    provenance: SubmitProvenance
    process: ProcessSettings


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

job_result_adapter: TypeAdapter[JobResult] = TypeAdapter(JobResult)


class HelloMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    worker: str
    backend: str
    resources: ResourceRequest


def coordinator_url(*, host: str, port: int, auth_token: str) -> str:
    return f"ws://furu:{auth_token}@{host}:{port}"
