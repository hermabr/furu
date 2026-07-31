from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

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


type LeaseJobResponse = Job | Literal["wait", "stop"]


class HelloMessage(BaseModel):
    """First message on a connection: the worker introduces itself."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["hello"] = "hello"
    worker: str
    backend: str
    resources: ResourceRequest


class ResultMessage(BaseModel):
    """Outcome of the worker's current assignment, covering all its members."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["result"] = "result"
    result: JobResultRequest


class WelcomeMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["welcome"] = "welcome"
    executor_id: str


class AssignMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["assign"] = "assign"
    job: Job


class StopMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["stop"] = "stop"
    reason: str


type WorkerMessage = Annotated[
    HelloMessage | ResultMessage,
    Field(discriminator="kind"),
]

type ServerMessage = Annotated[
    WelcomeMessage | AssignMessage | StopMessage,
    Field(discriminator="kind"),
]

worker_message_adapter: TypeAdapter[WorkerMessage] = TypeAdapter(WorkerMessage)
server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
