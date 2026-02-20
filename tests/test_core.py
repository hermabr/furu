from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from furu import Furu


def test_subclass_is_dataclass_kw_only_and_frozen():
    class Add(Furu[int]):
        a: int
        b: int

    assert is_dataclass(Add)
    sig = inspect.signature(Add)
    assert sig.parameters["a"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["b"].kind is inspect.Parameter.KEYWORD_ONLY
    obj = Add(a=1, b=2)
    assert obj.a == 1
    assert obj.b == 2
    with pytest.raises(TypeError):
        type.__call__(Add, 1, 2)
    with pytest.raises(FrozenInstanceError):
        setattr(obj, "a", 3)


def test_inheritance_includes_parent_and_child_fields():
    class Node(Furu[int]):
        name: str

    class WeightedNode(Node):
        weight: float

    obj = WeightedNode(name="x", weight=1.5)
    assert obj.name == "x"
    assert obj.weight == 1.5
