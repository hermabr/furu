from dataclasses import dataclass

type MinMax = tuple[int | None, int | None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequirements:
    cpus: MinMax | None = None
    memory: MinMax | None = None
    gpus: MinMax | None = None
