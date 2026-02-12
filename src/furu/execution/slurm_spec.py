from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, TypeAlias, cast


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

    @cached_property
    def _spec_key(self) -> str:
        return _build_slurm_spec_key(self)

    def __hash__(self) -> int:
        return hash(self._spec_key)


SlurmExecutorChoice: TypeAlias = SlurmSpec | Sequence[SlurmSpec]


class _ExecutorNode(Protocol):
    furu_hash: str

    def _executor(self) -> SlurmExecutorChoice: ...


def resolve_executor_specs(node: _ExecutorNode) -> tuple[SlurmSpec, ...]:
    specs = node._executor()
    if isinstance(specs, SlurmSpec):
        return (specs,)
    if isinstance(specs, Sequence):
        resolved_specs: list[SlurmSpec] = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, SlurmSpec):
                raise TypeError(
                    "Furu._executor() sequence entries must be SlurmSpec for "
                    f"{node.__class__.__name__} ({node.furu_hash}), "
                    f"got {type(spec).__name__} at index {index}."
                )
            resolved_specs.append(spec)
        if resolved_specs:
            return tuple(resolved_specs)
        raise TypeError(
            "Furu._executor() must return SlurmSpec or a non-empty sequence of "
            f"SlurmSpec for {node.__class__.__name__} ({node.furu_hash})."
        )
    raise TypeError(
        "Furu._executor() must return SlurmSpec or a sequence of SlurmSpec for "
        f"{node.__class__.__name__} ({node.furu_hash}), got {type(specs).__name__}."
    )


def resolve_executor_spec(node: _ExecutorNode) -> SlurmSpec:
    return resolve_executor_specs(node)[0]


def _normalize_extra(value: SlurmSpecExtraValue) -> SlurmSpecPayloadValue:
    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[str, SlurmSpecExtraValue], value)
        normalized: dict[str, SlurmSpecPayloadValue] = {}
        for key in sorted(mapping_value):
            normalized[key] = _normalize_extra(mapping_value[key])
        return normalized
    return value


def _build_slurm_spec_key(spec: SlurmSpec) -> str:
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


def slurm_spec_key(spec: SlurmSpec) -> str:
    return spec._spec_key
