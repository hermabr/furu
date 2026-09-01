from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple

import pytest
from pydantic import BaseModel

import furu
from furu import Missing, Spec
from furu.logging import _scoped_log_files
from furu.worker.backends.local import LocalThreadWorkerBackend


class Node(Spec[str]):
    name: str

    def create(self) -> str:
        return f"Node({self.name})"


class Counted(Spec[str]):
    name: str
    calls: ClassVar[list[str]] = []

    def create(self) -> str:
        self.calls.append(self.name)
        return f"Counted({self.name})"


class Pair(NamedTuple):
    a: Node
    b: Node


@dataclass(frozen=True)
class Bundle:
    train: Node
    evals: list[Node]


class Model(BaseModel):
    main: Node
    extra: dict[str, Node]


def test_create_dict_of_lists_returns_same_shape() -> None:
    out = furu.create(
        {"train": Node(name="t"), "evals": [Node(name=f"e{i}") for i in range(2)]}
    )

    assert out == {"train": "Node(t)", "evals": ["Node(e0)", "Node(e1)"]}


def test_create_tuple_namedtuple_dataclass_pydantic_shapes() -> None:
    assert furu.create((Node(name="x"), Node(name="y"))) == ("Node(x)", "Node(y)")

    pair = furu.create(Pair(a=Node(name="a"), b=Node(name="b")))
    assert isinstance(pair, Pair)
    assert pair == Pair(a="Node(a)", b="Node(b)")  # ty: ignore[invalid-argument-type]

    bundle = furu.create(Bundle(train=Node(name="t"), evals=[Node(name="e")]))
    assert isinstance(bundle, Bundle)
    assert bundle.train == "Node(t)"
    assert bundle.evals == ["Node(e)"]

    model = furu.create(Model(main=Node(name="m"), extra={"k": Node(name="k")}))
    assert isinstance(model, Model)
    assert model.main == "Node(m)"
    assert model.extra == {"k": "Node(k)"}


def test_create_single_spec_and_empty_list() -> None:
    assert furu.create(Node(name="single")) == "Node(single)"
    assert furu.create([]) == []
    assert furu.create({}) == {}


def test_create_duplicate_leaf_computed_once() -> None:
    Counted.calls.clear()
    out = furu.create({"x": Counted(name="dup"), "y": Counted(name="dup")})

    assert out == {"x": "Counted(dup)", "y": "Counted(dup)"}
    assert Counted.calls == ["dup"]


def test_create_rejects_non_spec_leaf_with_path() -> None:
    with pytest.raises(TypeError, match=r"got int at tree\['evals'\]\[1\]"):
        furu.create({"evals": [Node(name="ok"), 3]})


def test_create_rejects_sets() -> None:
    with pytest.raises(TypeError, match=r"sets are not supported .* at tree\['s'\]"):
        furu.create({"s": {Node(name="a")}})
    with pytest.raises(TypeError, match=r"sets are not supported .* at tree$"):
        furu.create(frozenset({Node(name="a")}))


def test_load_existing_same_shape_and_missing(tmp_path: Path) -> None:
    tree = {"train": Node(name="lt"), "evals": (Node(name="le0"), Node(name="le1"))}
    with pytest.raises(Missing, match=r"load_existing\(\) could not find a result"):
        furu.load_existing(tree)

    assert furu.create(tree) == {
        "train": "Node(lt)",
        "evals": ("Node(le0)", "Node(le1)"),
    }

    log_path = tmp_path / "load.log"
    with _scoped_log_files((log_path,)):
        assert furu.load_existing(tree) == {
            "train": "Node(lt)",
            "evals": ("Node(le0)", "Node(le1)"),
        }
    assert "loaded 3 furu objects" in log_path.read_text(encoding="utf-8")


def test_create_tree_on_local_backend() -> None:
    tree = {"a": Node(name="on-a"), "b": [Node(name="on-b"), Node(name="on-a")]}

    out = furu.create(tree, on=[LocalThreadWorkerBackend(max_workers=2)])

    assert out == {"a": "Node(on-a)", "b": ["Node(on-b)", "Node(on-a)"]}
