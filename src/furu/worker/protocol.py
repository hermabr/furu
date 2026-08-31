from __future__ import annotations

from pathlib import Path
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

    kind: Literal["job"] = "job"
    artifacts: list[ArtifactSpec]
    provenance: SubmitProvenance
    process: ProcessSettings


class CancelMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["cancel"] = "cancel"


type ServerMessage = Annotated[Job | CancelMessage, Field(discriminator="kind")]

server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


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

    kind: Literal["hello"] = "hello"
    worker: str
    backend: str
    resources: ResourceRequest
    running: list[ArtifactSpec] = Field(default_factory=list)


class TakeoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["takeover"] = "takeover"
    executor_id: str
    pool_keys: list[str]


type FirstMessage = Annotated[
    HelloMessage | TakeoverRequest, Field(discriminator="kind")
]

first_message_adapter: TypeAdapter[FirstMessage] = TypeAdapter(FirstMessage)


class PoolHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job_ids: list[str] = Field(default_factory=list)
    worker_files: list[Path] = Field(default_factory=list)


class TakeoverAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["accepted"] = "accepted"
    handoffs: dict[str, PoolHandoff]


class TakeoverRefused(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["refused"] = "refused"
    reason: str


type TakeoverResponse = Annotated[
    TakeoverAccepted | TakeoverRefused, Field(discriminator="kind")
]

takeover_response_adapter: TypeAdapter[TakeoverResponse] = TypeAdapter(TakeoverResponse)


def coordinator_url(*, host: str, port: int, auth_token: str) -> str:
    return f"ws://furu:{auth_token}@{host}:{port}"
