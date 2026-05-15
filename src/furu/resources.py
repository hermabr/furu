from dataclasses import dataclass

type Bounds = tuple[int | None, int | None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequirements:
    cpus: Bounds | None = None
    memory: Bounds | None = None
    gpus: Bounds | None = None
