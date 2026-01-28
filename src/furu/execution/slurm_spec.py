from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


SlurmSpecValue = str | int | float | bool
SlurmSpecExtraValue = SlurmSpecValue | Mapping[str, "SlurmSpecExtraValue"]


@dataclass(frozen=True)
class SlurmSpec:
    partition: str | None = None
    gpus: int = 0
    cpus: int = 4
    mem_gb: int = 16
    time_min: int = 60
    extra: Mapping[str, SlurmSpecExtraValue] | None = None


class _SpecNode(Protocol):
    furu_hash: str

    def _executor_spec_key(self) -> str: ...


def resolve_slurm_spec(specs: Mapping[str, SlurmSpec], node: _SpecNode) -> SlurmSpec:
    if "default" not in specs:
        raise KeyError("Missing slurm spec for key 'default'.")

    spec_key = node._executor_spec_key()
    if spec_key not in specs:
        raise KeyError(
            "Missing slurm spec for key "
            f"'{spec_key}' for node {node.__class__.__name__} ({node.furu_hash})."
        )

    return specs[spec_key]
