import math
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Literal

from furu.config import CodeVersion, get_config

type Reuse = Literal["never", "same_environment", "same_environment_same_spec"]


@dataclass(frozen=True, slots=True)
class Throttle:
    """A limit shared by every spec that names it, of any class and script.

    ``max_running`` caps what runs at once, ``max_starts`` what starts within
    any ``per_seconds``. ``counts`` says what a create() call counts as: one
    job, or each of its specs. Rules with one name share one count, so a
    per-minute and a per-hour limit on the same API are two rules.

    Enforced by the manager in memory only (ignored by in-process ``create()``,
    and reset if the manager restarts). A name's rate window sees starts only
    after a rate rule for that name is submitted, and there is no fairness
    guarantee between jobs sharing a name.
    """

    name: str
    _: KW_ONLY
    max_running: int | None = None
    max_starts: int | None = None
    per_seconds: float | None = None
    counts: Literal["jobs", "specs"] = "jobs"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a throttle needs a name, got {self.name!r}")
        for limit in ("max_running", "max_starts"):
            value = getattr(self, limit)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{limit} must be a positive int, got {value!r}")
        if (self.max_starts is None) != (self.per_seconds is None):
            raise ValueError("max_starts and per_seconds go together")
        if self.max_running is None and self.max_starts is None:
            raise ValueError(f"throttle {self.name!r} needs max_running or max_starts")
        if (per := self.per_seconds) is not None:
            # Coerce int to float so 60 and 60.0 compare and serialize the same.
            try:
                seconds = float(per) if type(per) in (int, float) else math.nan
            except OverflowError:
                seconds = math.inf
            if not 0 < seconds < math.inf:
                raise ValueError(f"per_seconds must be positive, got {per!r}")
            object.__setattr__(self, "per_seconds", seconds)
        if self.counts not in ("jobs", "specs"):
            raise ValueError(f"counts must be 'jobs' or 'specs', got {self.counts!r}")


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
    code_version "traced" ties the result to the repo code its create() ran;
    the default comes from [tool.furu] code_version (or FURU_CODE_VERSION).
    throttles are the limits its jobs start under, in any order.
    """

    storage: Path = field(
        default_factory=lambda: get_config().directories.anchored().objects
    )
    environment: Mapping[str, str | None] = field(default_factory=dict)
    required_environment_variables: tuple[str, ...] = ()
    reuse: Reuse = "same_environment"
    code_version: CodeVersion = field(default_factory=lambda: get_config().code_version)
    throttles: tuple[Throttle, ...] = ()

    def __post_init__(self) -> None:
        for rule in self.throttles:
            if not isinstance(rule, Throttle):
                raise TypeError(f"throttles must contain Throttle, got {rule!r}")
        throttles = sorted(
            dict.fromkeys(self.throttles),
            key=lambda rule: (
                rule.name,
                rule.counts,
                rule.max_running or 0,
                rule.max_starts or 0,
                rule.per_seconds or 0.0,
            ),
        )
        # Sorted by name, then counts: two that clash are neighbours.
        for one, other in pairwise(throttles):
            if one.name == other.name and one.counts != other.counts:
                raise ValueError(f"throttle {one.name!r} counts both jobs and specs")
        object.__setattr__(self, "throttles", tuple(throttles))
