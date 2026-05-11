from __future__ import annotations

import shutil
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from furu.core import Furu
    from furu.graph import ArtifactNode

_worker_execution_active: ContextVar[bool] = ContextVar(
    "_worker_execution_active",
    default=False,
)


def in_worker_execution_context() -> bool:
    return _worker_execution_active.get()


@contextmanager
def worker_execution_context() -> Iterator[None]:
    token = _worker_execution_active.set(True)
    try:
        yield
    finally:
        _worker_execution_active.reset(token)


class _DependencyNotReady(BaseException):
    def __init__(
        self,
        dependencies: Sequence[Furu[Any]],
        *,
        call_kind: Literal["load_or_create", "try_load"],
    ) -> None:
        from furu.graph import ArtifactNode, node_key_for
        from furu.metadata import artifact_spec_for

        self.dependencies = tuple(dependencies)
        self.call_kind = call_kind
        self.nodes: tuple[ArtifactNode, ...] = tuple(
            ArtifactNode(
                key=node_key_for(dep),
                artifact=artifact_spec_for(dep),
            )
            for dep in self.dependencies
        )

        super().__init__(
            f"{call_kind} discovered {len(self.dependencies)} missing dependency/dependencies"
        )


def cleanup_incomplete_attempt(obj: Furu[Any]) -> None:
    from furu.migration import result_dir_for_loading

    if result_dir_for_loading(obj) is not None:
        return

    if obj.data_dir.exists():
        shutil.rmtree(obj.data_dir, ignore_errors=True)


def execute_one_artifact(lease_node: ArtifactNode) -> Any:
    from furu.core import Furu
    from furu.execution import _execute_group
    from furu.graph import node_key_for
    from furu.locking import lock_many
    from furu.migration import result_dir_for_loading
    from furu.server import (
        LeaseDependencyNotReady,
        LeaseDone,
        LeaseFailed,
    )
    from furu.storage_layout import (
        compute_lock_path_in,
        internal_furu_dir_in,
    )

    obj = Furu.from_artifact(lease_node.artifact)

    if node_key_for(obj) != lease_node.key:
        return LeaseFailed(
            kind="failed",
            node_key=lease_node.key,
            error_type="NodeKeyMismatch",
            error_message=(
                "Artifact reconstructed to a different data path. "
                "Check that the scheduler and worker share the same FURU_DIRECTORIES__DATA."
            ),
            traceback="",
        )

    if result_dir_for_loading(obj) is not None:
        return LeaseDone(
            kind="done",
            node_key=lease_node.key,
        )

    try:
        internal_furu_dir_in(obj.data_dir).mkdir(parents=True, exist_ok=True)

        with lock_many([compute_lock_path_in(obj.data_dir)]) as has_lock:
            if result_dir_for_loading(obj) is not None:
                return LeaseDone(
                    kind="done",
                    node_key=lease_node.key,
                )

            results_by_dir: dict[Any, Any] = {}

            with worker_execution_context():
                _execute_group(
                    [obj],
                    has_lock=has_lock,
                    results_by_dir=results_by_dir,
                )

        return LeaseDone(
            kind="done",
            node_key=lease_node.key,
        )

    except _DependencyNotReady as exc:
        cleanup_incomplete_attempt(obj)

        return LeaseDependencyNotReady(
            kind="dependency_not_ready",
            blocked=lease_node.key,
            call_kind=exc.call_kind,
            dependencies=exc.nodes,
        )

    except BaseException as exc:
        return LeaseFailed(
            kind="failed",
            node_key=lease_node.key,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )
