from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from furu.graph import GraphFragment, NodeKey

if TYPE_CHECKING:
    from furu.submission import Submission


class Executor(Protocol):
    def submit(
        self,
        *,
        graph: GraphFragment,
        roots: Sequence[NodeKey],
        input_order: Sequence[NodeKey],
        single_input: bool,
    ) -> Submission[Any]: ...
