from dataclasses import dataclass

from pydantic import BaseModel

type ResourceConstraint = tuple[int | None, int | None] | None


class ResourceRequest(BaseModel):
    cpus: int = 1
    gpus: int = 0
    memory: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequirements:
    cpus: ResourceConstraint = None
    memory: ResourceConstraint = None
    gpus: ResourceConstraint = None
