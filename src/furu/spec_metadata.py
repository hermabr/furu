from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from furu.config import get_config


@dataclass(frozen=True, slots=True, order=True)
class GiB:
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError(f"GiB count must be non-negative, got {self.count}")


@dataclass(frozen=True, slots=True)
class Between[T]:
    low: T
    high: T | None

    def __post_init__(self) -> None:
        if isinstance(self.low, int) and self.low < 0:
            raise ValueError(f"range lower bound must be non-negative, got {self.low}")
        if self.high is not None and cast(Any, self.high) < self.low:
            raise ValueError(
                f"range upper bound {self.high} is below lower bound {self.low}"
            )


def between[T](low: T, high: T) -> Between[T]:
    return Between(low, high)


def at_least[T](minimum: T) -> Between[T]:
    return Between(minimum, None)


@dataclass(frozen=True, slots=True, kw_only=True)
class Requires:
    gpus: int | Between[int] | None = None
    cpus: int | Between[int] | None = None
    ram: GiB | Between[GiB] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Throttle:
    max_running: int

    def __post_init__(self) -> None:
        if self.max_running < 1:
            raise ValueError(f"max_running must be positive, got {self.max_running}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Subprocess:
    """Run create() in a child Python process owned by the worker.

    A None value in environment removes the variable from the child, as
    opposed to setting it to the empty string.

    Values in required_environment must exist in the effective child process
    environment. They are checked by name only; values are never logged.
    """

    environment: Mapping[str, str | None] = field(default_factory=dict)
    required_environment: tuple[str, ...] = ()
    reuse: Literal["never", "same_environment", "same_environment_same_spec"] = (
        "same_environment"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Metadata:
    storage: Path = field(default_factory=lambda: get_config().run_directories.objects)
    requires: Requires = Requires()
    execution: Literal["inline"] | Subprocess = "inline"
