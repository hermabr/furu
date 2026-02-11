from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Protocol, TypeAlias, cast


SlurmSpecValue = str | int | float | bool
SlurmSpecExtraValue = SlurmSpecValue | Mapping[str, "SlurmSpecExtraValue"]
SlurmSpecPayloadValue: TypeAlias = (
    SlurmSpecValue | None | dict[str, "SlurmSpecPayloadValue"]
)


@dataclass(frozen=True)
class SlurmSpec:
    partition: str | None = None
    gpus: int = 0
    cpus: int = 4
    mem_gb: int = 16
    time_min: int = 60
    extra: Mapping[str, SlurmSpecExtraValue] | None = None


class _ExecutorNode(Protocol):
    furu_hash: str

    def _executor(self) -> SlurmSpec: ...


def resolve_executor_spec(node: _ExecutorNode) -> SlurmSpec:
    spec = node._executor()
    if isinstance(spec, SlurmSpec):
        return spec
    raise TypeError(
        "Furu._executor() must return SlurmSpec for "
        f"{node.__class__.__name__} ({node.furu_hash}), got {type(spec).__name__}."
    )


def _normalize_extra(value: SlurmSpecExtraValue) -> SlurmSpecPayloadValue:
    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[str, SlurmSpecExtraValue], value)
        normalized: dict[str, SlurmSpecPayloadValue] = {}
        for key in sorted(mapping_value):
            normalized[key] = _normalize_extra(mapping_value[key])
        return normalized
    return value


def slurm_spec_key(spec: SlurmSpec) -> str:
    payload: dict[str, SlurmSpecPayloadValue] = {
        "partition": spec.partition,
        "gpus": spec.gpus,
        "cpus": spec.cpus,
        "mem_gb": spec.mem_gb,
        "time_min": spec.time_min,
        "extra": _normalize_extra(spec.extra) if spec.extra is not None else {},
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]

    partition = spec.partition or "default"
    safe_partition = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in partition.lower()
    )
    safe_partition = safe_partition.strip("-") or "default"
    return f"{safe_partition}-{digest}"
