"""Every message between scripts, the manager and workers.

A connection's first message says who is calling: a script sends ``submit``,
a worker sends ``hello``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from furu.config import _Config
from furu.logging import _group_label
from furu.snapshot import CodeLocation
from furu.spec_metadata import Metadata, Reuse
from furu.storage._layout import attempt_dir_in, log_path_in, run_name
from furu.utils import spec_label


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProcessSettings(_Message):
    environment: dict[str, str | None]
    required_environment_variables: tuple[str, ...]
    reuse: Reuse

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> ProcessSettings:
        return cls(
            environment=dict(metadata.environment),
            required_environment_variables=metadata.required_environment_variables,
            reuse=metadata.reuse,
        )


class Job(_Message):
    kind: Literal["job"] = "job"
    # One create() call: the ids of the specs of one batch in row order, or of
    # one spec.
    specs: list[str]
    # Their plan lines and the lines their fields reach, so the worker builds
    # them without reading a plan file or hashing anything.
    lines: list[str]
    # The plan file that lists them.
    plan: Path
    # Their class folder, which the worker cannot compute without importing
    # the spec class.
    class_dir: Path
    execution_log: Path
    process: ProcessSettings
    # The submitting repo's settings (absolute data directories) and code, so
    # a worker serving many repos stores each result in the repo it came from.
    config: _Config
    code: CodeLocation
    # The stored names of the classes the launch asked about. A spec of
    # another class that a create() builds is checked where it is built.
    asked: frozenset[str] = frozenset()

    @property
    def label(self) -> str:
        return _group_label(spec_label(*self.specs[0].split(":")), len(self.specs))

    @property
    def log(self) -> Path:
        """Where its create() writes: the log in the working folder of its run."""
        run = run_name([spec.rpartition(":")[2] for spec in self.specs])
        return log_path_in(attempt_dir_in(self.class_dir, run))


class CancelMessage(_Message):
    kind: Literal["cancel"] = "cancel"


class JobCompletedResult(_Message):
    status: Literal["completed"] = "completed"


class JobFailedResult(_Message):
    status: Literal["failed"] = "failed"
    error: str
    # A traced file changed after the child imported it: retry in a fresh child.
    stale_code: bool = False


class JobBlockedResult(_Message):
    status: Literal["blocked"] = "blocked"
    # The ids of the specs its create() asked for that are not stored, the
    # lines that describe them, and the plan file the lines were saved in.
    dependencies: list[str]
    lines: list[str]
    plan: Path


type JobResult = Annotated[
    JobCompletedResult | JobFailedResult | JobBlockedResult,
    Field(discriminator="status"),
]

job_result_adapter: TypeAdapter[JobResult] = TypeAdapter(JobResult)


# --- worker <-> manager --------------------------------------------------------


class HelloMessage(_Message):
    kind: Literal["hello"] = "hello"
    worker: str
    backend: str
    pool: str
    log: Path | None = None  # the worker's own log file, if it has one


type ServerMessage = Annotated[Job | CancelMessage, Field(discriminator="kind")]

server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


# --- script <-> manager --------------------------------------------------------


class SlurmPool(_Message):
    """How the manager starts a worker; identical recipes share one pool."""

    kind: Literal["slurm"] = "slurm"
    max_workers: int
    sbatch_args: list[str]
    pre_worker_commands: list[str]
    use_job_arrays: bool
    job_name: str
    idle_timeout_seconds: float


class LocalPool(_Message):
    """Threads in the script's own process, against its in-process manager."""

    kind: Literal["local"] = "local"
    max_workers: int


type Pool = Annotated[SlurmPool | LocalPool, Field(discriminator="kind")]


class Batch(_Message):
    batch: str  # jobs of one script with the same id may share a worker call
    cap: int


class JobThrottle(_Message):
    key: str
    max_running: int


class Submission(_Message):
    job: Job
    # The pools that can run this job; None runs it alone.
    pools: dict[str, Batch | None]
    throttle: JobThrottle | None


class SubmitMessage(_Message):
    kind: Literal["submit"] = "submit"
    pools: dict[str, Pool]
    jobs: list[Submission]


type ScriptMessage = Annotated[
    SubmitMessage | CancelMessage, Field(discriminator="kind")
]

script_message_adapter: TypeAdapter[ScriptMessage] = TypeAdapter(ScriptMessage)


class ResultMessage(_Message):
    kind: Literal["result"] = "result"
    object_ids: list[str]
    result: JobResult


class WarningMessage(_Message):
    kind: Literal["warning"] = "warning"
    reason: str


class ErrorMessage(_Message):
    kind: Literal["error"] = "error"
    reason: str


type ManagerMessage = Annotated[
    ResultMessage | WarningMessage | ErrorMessage, Field(discriminator="kind")
]

manager_message_adapter: TypeAdapter[ManagerMessage] = TypeAdapter(ManagerMessage)


type FirstMessage = Annotated[SubmitMessage | HelloMessage, Field(discriminator="kind")]

first_message_adapter: TypeAdapter[FirstMessage] = TypeAdapter(FirstMessage)
