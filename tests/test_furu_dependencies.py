import json

import chz
import pytest

import furu
from furu import DependencyChzSpec, DependencySpec, furu_dep


class DependencyTask(furu.Furu[int]):
    value: int

    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(self.value))
        return self.value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


@chz.chz
class DependencyGroup(DependencyChzSpec):
    tasks: list[DependencyTask]


class DependencyCollection(furu.Furu[int]):
    n_tasks: int
    base_task: DependencyTask

    @furu_dep
    def dynamic_dependencies(self) -> DependencySpec:
        return DependencyGroup(
            tasks=[DependencyTask(value=i) for i in range(self.n_tasks)]
        )

    def _create(self) -> int:
        value = self.n_tasks
        (self.furu_dir / "collection.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "collection.json").read_text())


class DependencyLayerTwo(furu.Furu[int]):
    leaf: DependencyTask

    @furu_dep
    def extra_dependencies(self) -> DependencySpec:
        return [DependencyTask(value=2)]

    def _create(self) -> int:
        value = self.leaf.value
        (self.furu_dir / "layer_two.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "layer_two.json").read_text())


class DependencyLayerOne(furu.Furu[int]):
    layer_two: DependencyLayerTwo

    @furu_dep
    def extra_dependencies(self) -> DependencySpec:
        return [DependencyTask(value=3)]

    def _create(self) -> int:
        value = self.layer_two.leaf.value
        (self.furu_dir / "layer_one.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "layer_one.json").read_text())


class DuplicateDependencyHolder(furu.Furu[int]):
    @furu_dep
    def duplicate_dependencies(self) -> DependencySpec:
        return [DependencyTask(value=1), DependencyTask(value=1)]

    def _create(self) -> int:
        value = 1
        (self.furu_dir / "duplicate.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "duplicate.json").read_text())


class AliasDecoratorHolder(furu.Furu[int]):
    @furu_dep
    def alias_dependencies(self) -> DependencySpec:
        return [DependencyTask(value=7)]

    def _create(self) -> int:
        value = 7
        (self.furu_dir / "alias.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "alias.json").read_text())


@chz.chz
class BadDependencySpec(DependencyChzSpec):
    tasks: list[DependencyTask]
    label: str


class BadDependencyCollection(furu.Furu[int]):
    @furu_dep
    def bad_dependencies(self) -> DependencySpec:
        return BadDependencySpec(tasks=[DependencyTask(value=1)], label="bad")

    def _create(self) -> int:
        value = 0
        (self.furu_dir / "bad.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "bad.json").read_text())


class Fibonacci(furu.Furu[int]):
    n: int

    @furu_dep
    def recurrence_dependencies(self) -> DependencySpec:
        if self.n <= 1:
            return []
        return [Fibonacci(n=self.n - 1), Fibonacci(n=self.n - 2)]

    def _create(self) -> int:
        value = self.n
        (self.furu_dir / "fib.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "fib.json").read_text())


def test_dependencies_deduplicates_fields_and_dependencies(furu_tmp_root) -> None:
    base_task = DependencyTask(value=0)
    collection = DependencyCollection(n_tasks=2, base_task=base_task)

    deps = collection._get_dependencies(recursive=False)

    assert deps[0] is base_task
    values = [dep.value for dep in deps if isinstance(dep, DependencyTask)]
    assert values == [0, 1]


def test_dependencies_deduplicates_repeated_dependencies(furu_tmp_root) -> None:
    deps = DuplicateDependencyHolder()._get_dependencies(recursive=False)

    values = [dep.value for dep in deps if isinstance(dep, DependencyTask)]
    assert values == [1]


def test_furu_dep_marks_dependency(furu_tmp_root) -> None:
    deps = AliasDecoratorHolder()._get_dependencies(recursive=False)

    values = [dep.value for dep in deps if isinstance(dep, DependencyTask)]
    assert values == [7]


def test_dependencies_two_layers_with_dependencies(furu_tmp_root) -> None:
    leaf = DependencyTask(value=1)
    layer_two = DependencyLayerTwo(leaf=leaf)
    root = DependencyLayerOne(layer_two=layer_two)

    deps = root._get_dependencies()

    assert deps[0] is layer_two
    assert deps[1] is leaf
    task_values = [dep.value for dep in deps if isinstance(dep, DependencyTask)]
    assert task_values == [1, 2, 3]


def test_dependencies_fibonacci_recursive(furu_tmp_root) -> None:
    deps = Fibonacci(n=4)._get_dependencies()

    values = [dep.n for dep in deps if isinstance(dep, Fibonacci)]
    assert values == [3, 2, 1, 0]


def test_dependencies_fibonacci_direct(furu_tmp_root) -> None:
    deps = Fibonacci(n=4)._get_dependencies(recursive=False)

    values = [dep.n for dep in deps if isinstance(dep, Fibonacci)]
    assert values == [3, 2]


def test_dependencies_spec_rejects_non_furu_fields(furu_tmp_root) -> None:
    with pytest.raises(TypeError, match="label"):
        BadDependencyCollection()._get_dependencies()


def test_hash_includes_dependency_spec(monkeypatch) -> None:
    collection = DependencyCollection(n_tasks=1, base_task=DependencyTask(value=0))
    original_hash = collection.furu_hash

    @furu_dep
    def alt_dependencies(self) -> DependencySpec:
        return [DependencyTask(value=99)]

    monkeypatch.setattr(DependencyCollection, "dynamic_dependencies", alt_dependencies)

    assert collection.furu_hash == original_hash

    refreshed = DependencyCollection(n_tasks=1, base_task=DependencyTask(value=0))
    assert refreshed.furu_hash != original_hash


def test_hash_ignores_nested_dependencies() -> None:
    leaf = DependencyTask(value=1)
    layer_two = DependencyLayerTwo(leaf=leaf)
    root = DependencyLayerOne(layer_two=layer_two)

    hashes = root._dependency_hashes()
    expected = sorted({layer_two.furu_hash, DependencyTask(value=3).furu_hash})

    assert hashes == expected


def test_hash_ignores_duplicate_dependencies(monkeypatch) -> None:
    holder = DuplicateDependencyHolder()
    original_hash = holder.furu_hash

    @furu_dep
    def unique_dependencies(self) -> DependencySpec:
        return [DependencyTask(value=1)]

    monkeypatch.setattr(
        DuplicateDependencyHolder,
        "duplicate_dependencies",
        unique_dependencies,
    )

    assert holder.furu_hash == original_hash
