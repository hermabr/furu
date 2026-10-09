from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from furu.config import get_config

type Reuse = Literal["never", "same_environment", "same_environment_same_spec"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Throttle:
    max_running: int

    def __post_init__(self) -> None:
        if self.max_running < 1:
            raise ValueError(f"max_running must be positive, got {self.max_running}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Metadata:
    """Where a spec stores results and how the worker runs its create().

    For storage inside your repo, use furu.submitting_repo_root(), not
    __file__: workers run code from a snapshot.

    Workers run create() in a child Python process. A None value in
    environment removes the variable from the child, as opposed to setting it
    to the empty string. Variables named in required_environment_variables (e.g.
    HF_TOKEN) must be set in the child environment; the job fails before
    spawning otherwise. reuse controls when a warm child is kept between jobs.
    """

    storage: Path = field(default_factory=lambda: get_config().run_directories.objects)
    environment: Mapping[str, str | None] = field(default_factory=dict)
    required_environment_variables: tuple[str, ...] = ()
    reuse: Reuse = "same_environment"
