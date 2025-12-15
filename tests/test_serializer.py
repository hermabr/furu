import datetime
import importlib
import pathlib
from pathlib import Path
from typing import Any

import chz
import pytest

import huldra


class Foo:
    a: int = chz.field()
    p: Path = chz.field()
    _private: int = chz.field(default=0)


type.__setattr__(
    Foo,
    "__annotations__",
    {"a": int, "p": Path, "_private": int},
)
Foo: Any = chz.chz(Foo)


def test_get_classname_rejects_main_module() -> None:
    MainLike = type("MainLike", (), {})
    MainLike.__module__ = "__main__"
    with pytest.raises(ValueError, match="__main__"):
        huldra.HuldraSerializer.get_classname(MainLike())


def test_to_dict_from_dict_roundtrip() -> None:
    obj = Foo(a=1, p=Path("x/y"), _private=7)
    data = huldra.HuldraSerializer.to_dict(obj)
    obj2 = huldra.HuldraSerializer.from_dict(data)
    assert obj2 == obj


def test_compute_hash_ignores_private_fields() -> None:
    a = Foo(a=1, p=Path("x/y"), _private=1)
    b = Foo(a=1, p=Path("x/y"), _private=999)
    assert huldra.HuldraSerializer.compute_hash(a) == huldra.HuldraSerializer.compute_hash(
        b
    )


def test_to_python_is_evaluable() -> None:
    obj = Foo(a=3, p=Path("a/b"))
    code = huldra.HuldraSerializer.to_python(obj, multiline=False)

    mod = importlib.import_module(obj.__class__.__module__)
    env = {"pathlib": pathlib, "datetime": datetime}
    env.update(mod.__dict__)
    env[obj.__class__.__module__] = mod
    obj2 = eval(code, env)
    assert obj2 == obj


def test_missing_is_not_serializable() -> None:
    with pytest.raises(ValueError, match="MISSING"):
        huldra.HuldraSerializer.to_dict(huldra.MISSING)
