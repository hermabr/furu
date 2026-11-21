from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from furu.protocol.messages import LocalPool
from furu.resources import Worker
from furu.utils import _hash_dict_deterministically

if TYPE_CHECKING:
    from furu.core import Spec


@dataclass(frozen=True, slots=True)
class LocalThreadWorkerBackend:
    """Worker threads in this process, running the live worktree."""

    max_workers: int = 1
    worker: Worker = field(default_factory=Worker.here)
    accepts: Callable[[Spec], bool] | None = None

    @property
    def pool_key(self) -> str:
        return "local:" + _hash_dict_deterministically(
            {"worker": dataclasses.asdict(self.worker)}
        )

    def pool(self) -> LocalPool:
        return LocalPool(max_workers=self.max_workers)
