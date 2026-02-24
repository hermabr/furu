from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from furu.utils import JsonValue


class GitData(BaseModel):
    commit: str
    branch: str
    remote: str
    # patch: dict # TODO: add this
    # submodules: dict # TODO: add this


class Metadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        strict=True,
        revalidate_instances="always",
    )
    # python_def: str
    artifact: JsonValue
    artifact_hash: str
    schema_: JsonValue
    schema_hash: str
    data_path: Path
    started_at: datetime
    # git: GitData | None
    # command: list[str] # TODO: find/decide what the most elegant approach is here
    # hostname: str
    # user: str
    # pid: int
    # TODO: what other information should i record about the person starting this run?
    # furu_package_version: str
    # TODO: include some hardware information and machine info, such as os/version
    # python_version: str
    # executor info, such as local, local executor, slurm dag or slurm worker


class CompletedMetadata(Metadata):
    # traced_function_hashes: list[
    #     dict[str, str]
    # ]  # TODO: this probably means i don't really need to record the package versions? in particular since git data would have uv.lock and pyproject.toml already
    # slurm: SlurmData
    completed_at: datetime
