from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

type ResourceConstraint = tuple[int | None, int | None] | None


class ResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cpus: int = Field(default=1, ge=1)
    gpus: int = Field(default=0, ge=0)
    memory: int | None = Field(default=None, ge=0)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequirements:
    cpus: ResourceConstraint = None
    memory: ResourceConstraint = None
    gpus: ResourceConstraint = None


def satisfies_resource_requirements(
    requirements: ResourceRequirements | None,
    request: ResourceRequest,
) -> bool:
    if requirements is None:
        return True

    return (
        _constraint_satisfied(request.cpus, requirements.cpus)
        and _constraint_satisfied(request.gpus, requirements.gpus)
        and _constraint_satisfied(request.memory, requirements.memory)
    )


def _constraint_satisfied(
    value: int | None,
    constraint: ResourceConstraint,
) -> bool:
    if constraint is None:
        return True

    lower, upper = constraint
    if lower is None and upper is None:
        return True
    if value is None:
        return False
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True
