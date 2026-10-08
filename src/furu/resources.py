from __future__ import annotations

import glob
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Worker:
    """What one worker offers; specs pick workers with ``Spec.runs_on``.

    ``labels`` name hardware or placement facts that counts cannot express,
    e.g. ``("hopper",)`` for a pool constrained to H100 nodes.
    """

    cpus: int = 1
    gpus: int = 0
    memory_gib: int = 0
    labels: tuple[str, ...] = ()

    @classmethod
    def here(cls) -> Worker:
        """Describe this machine, honoring ``CUDA_VISIBLE_DEVICES``."""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        return cls(
            cpus=len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1,
            gpus=len(glob.glob("/dev/nvidia[0-9]*"))
            if visible is None
            else len([device for device in visible.split(",") if device]),
            memory_gib=os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") >> 30,
        )
