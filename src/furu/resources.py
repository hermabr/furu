from dataclasses import dataclass

type ResourceRange = tuple[int | None, int | None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequirements:
    cpus: ResourceRange | None = None
    memory: ResourceRange | None = None
    gpus: ResourceRange | None = None
