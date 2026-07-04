from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GiB:
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError(f"GiB count must be non-negative, got {self.count}")


@dataclass(frozen=True, slots=True)
class Between:
    low: int
    high: int | None

    def __post_init__(self) -> None:
        if self.low < 0:
            raise ValueError(f"range lower bound must be non-negative, got {self.low}")
        if self.high is not None and self.high < self.low:
            raise ValueError(
                f"range upper bound {self.high} is below lower bound {self.low}"
            )


def between(low: int, high: int) -> Between:
    return Between(low, high)


def at_least(minimum: int) -> Between:
    return Between(minimum, None)


@dataclass(frozen=True, slots=True, kw_only=True)
class Requires:
    gpus: int | Between = 0
    cpus: int | Between = 1
    ram: GiB | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Throttle:
    max_running: int

    def __post_init__(self) -> None:
        if self.max_running < 1:
            raise ValueError(f"max_running must be positive, got {self.max_running}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Metadata:
    storage: Path | None = None
    requires: Requires = Requires()
