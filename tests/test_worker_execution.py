from __future__ import annotations

import furu


class WorkerLeaf(furu.Furu[str]):
    name: str
    create_calls: int = 0

    def create(self) -> str:
        return f"leaf:{self.name}"


class WorkerParent(furu.Furu[str]):
    name: str

    def create(self) -> str:
        return WorkerLeaf(name=self.name).load_or_create()


class WorkerTryLoadParent(furu.Furu[str]):
    child: WorkerLeaf

    def create(self) -> str:
        return self.child.try_load()


def test_local_executor_submit_returns_single_result() -> None:
    result = WorkerLeaf(name="a").submit(executor=furu.LocalExecutor()).result()

    assert result == "leaf:a"


def test_local_executor_preserves_list_shape_and_order() -> None:
    result = furu.submit(
        [WorkerLeaf(name="a"), WorkerLeaf(name="b")],
        executor=furu.LocalExecutor(num_workers=2),
    ).result()

    assert result == ["leaf:a", "leaf:b"]


def test_local_executor_retries_parent_after_dynamic_load_or_create_dependency() -> (
    None
):
    result = WorkerParent(name="dynamic").submit(executor=furu.LocalExecutor()).result()

    assert result == "leaf:dynamic"


def test_local_executor_retries_parent_after_dynamic_try_load_dependency() -> None:
    child = WorkerLeaf(name="try")
    parent = WorkerTryLoadParent(child=child)

    result = parent.submit(executor=furu.LocalExecutor()).result()

    assert result == "leaf:try"
