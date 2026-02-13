import datetime
import importlib
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import chz
import pytest

import furu

if TYPE_CHECKING:

    class NotDefinedType: ...


class _FooBase:
    a: int = chz.field()
    p: Path = chz.field()
    _private: int = chz.field(default=0)


type.__setattr__(
    _FooBase,
    "__annotations__",
    {"a": int, "p": Path, "_private": int},
)
Foo = cast(Any, chz.chz(_FooBase))


@dataclass(frozen=True)
class FrozenDataclass:
    a: int
    p: Path
    _private: int = 0


@dataclass
class MutableDataclass:
    a: int
    p: Path
    _private: int = 0


@dataclass(frozen=True)
class NestedInnerDataclass:
    value: int
    _private: int = 0


@dataclass(frozen=True)
class NestedOuterDataclass:
    inner: NestedInnerDataclass
    p: Path


@dataclass(frozen=True)
class DataclassWithNonInitField:
    value: int
    derived: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "derived", self.value * 2)


@dataclass(frozen=True)
class RequiresTwoFields:
    first: int
    second: int


@dataclass(frozen=True)
class OuterWithNestedDataclass:
    inner: RequiresTwoFields


@dataclass(frozen=True)
class DataclassWithStringPathAnnotation:
    p: "Path"


@dataclass(frozen=True)
class DataclassWithUnresolvedTypeHint:
    p: "Path"
    missing: "NotDefinedType"


class RequiresTwoArgs:
    def __init__(self, first: int, second: int) -> None:
        self.first = first
        self.second = second


class RequiresKeywordOnlyArg:
    def __init__(self, first: int, *, second: int) -> None:
        self.first = first
        self.second = second


@dataclass(frozen=True)
class DataclassWithTypeErrorInPostInit:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise TypeError("post-init type error")


class HashValueFuru(furu.Furu[int]):
    value: int = chz.field()

    def _create(self) -> int:
        return self.value

    def _load(self) -> int:
        return self.value


@dataclass(frozen=True)
class DataclassWithFuru:
    dep: HashValueFuru
    _private: int = 0


@dataclass(frozen=True)
class NestedDataclassWithFuru:
    inner: DataclassWithFuru
    label: str


def test_get_classname_rejects_main_module() -> None:
    MainLike = type("MainLike", (), {})
    MainLike.__module__ = "__main__"
    with pytest.raises(ValueError, match="__main__"):
        furu.FuruSerializer.get_classname(MainLike())


def test_to_dict_from_dict_roundtrip() -> None:
    obj = Foo(a=1, p=Path("x/y"), _private=7)
    data = furu.FuruSerializer.to_dict(obj)
    obj2 = furu.FuruSerializer.from_dict(data)
    assert obj2 == obj


def test_compute_hash_ignores_private_fields() -> None:
    a = Foo(a=1, p=Path("x/y"), _private=1)
    b = Foo(a=1, p=Path("x/y"), _private=999)
    assert furu.FuruSerializer.compute_hash(a) == furu.FuruSerializer.compute_hash(b)


def test_hash_ignores_private_fields_in_config() -> None:
    obj = Foo(a=1, p=Path("x/y"), _private=7)
    data = furu.FuruSerializer.to_dict(obj)
    assert "_private" in data
    data["_private"] = 999
    assert furu.FuruSerializer.compute_hash(data) == furu.FuruSerializer.compute_hash(
        obj
    )


def test_to_python_is_evaluable() -> None:
    obj = Foo(a=3, p=Path("a/b"))
    code = furu.FuruSerializer.to_python(obj, multiline=False)

    mod = importlib.import_module(obj.__class__.__module__)
    env = {"pathlib": pathlib, "datetime": datetime}
    env.update(mod.__dict__)
    env[obj.__class__.__module__] = mod
    obj2 = eval(code, env)
    assert obj2 == obj


def test_missing_is_not_serializable() -> None:
    with pytest.raises(ValueError, match="MISSING"):
        furu.FuruSerializer.to_dict(furu.MISSING)


def test_to_dict_from_dict_roundtrip_plain_dataclass() -> None:
    obj = FrozenDataclass(a=1, p=Path("x/y"), _private=7)
    data = furu.FuruSerializer.to_dict(obj)
    obj2 = furu.FuruSerializer.from_dict(data)
    assert obj2 == obj


def test_compute_hash_supports_plain_dataclass() -> None:
    a = FrozenDataclass(a=1, p=Path("x/y"), _private=1)
    b = FrozenDataclass(a=1, p=Path("x/y"), _private=999)
    assert furu.FuruSerializer.compute_hash(a) == furu.FuruSerializer.compute_hash(b)


def test_compute_hash_supports_mutable_plain_dataclass() -> None:
    a = MutableDataclass(a=1, p=Path("x/y"), _private=1)
    b = MutableDataclass(a=1, p=Path("x/y"), _private=999)
    assert furu.FuruSerializer.compute_hash(a) == furu.FuruSerializer.compute_hash(b)


def test_to_dict_from_dict_roundtrip_nested_plain_dataclass() -> None:
    obj = NestedOuterDataclass(
        inner=NestedInnerDataclass(value=1, _private=7),
        p=Path("x/y"),
    )
    data = furu.FuruSerializer.to_dict(obj)
    obj2 = furu.FuruSerializer.from_dict(data)
    assert obj2 == obj


def test_compute_hash_supports_nested_plain_dataclass() -> None:
    a = NestedOuterDataclass(
        inner=NestedInnerDataclass(value=1, _private=1), p=Path("x/y")
    )
    b = NestedOuterDataclass(
        inner=NestedInnerDataclass(value=1, _private=999),
        p=Path("x/y"),
    )
    c = NestedOuterDataclass(
        inner=NestedInnerDataclass(value=2, _private=1), p=Path("x/y")
    )

    assert furu.FuruSerializer.compute_hash(a) == furu.FuruSerializer.compute_hash(b)
    assert furu.FuruSerializer.compute_hash(a) != furu.FuruSerializer.compute_hash(c)


def test_compute_hash_supports_nested_dataclass_with_furu() -> None:
    a = NestedDataclassWithFuru(
        inner=DataclassWithFuru(dep=HashValueFuru(value=1), _private=1),
        label="run",
    )
    b = NestedDataclassWithFuru(
        inner=DataclassWithFuru(dep=HashValueFuru(value=1), _private=999),
        label="run",
    )
    c = NestedDataclassWithFuru(
        inner=DataclassWithFuru(dep=HashValueFuru(value=2), _private=1),
        label="run",
    )

    assert furu.FuruSerializer.compute_hash(a) == furu.FuruSerializer.compute_hash(b)
    assert furu.FuruSerializer.compute_hash(a) != furu.FuruSerializer.compute_hash(c)

    data = furu.FuruSerializer.to_dict(a)
    obj2 = furu.FuruSerializer.from_dict(data)
    assert obj2 == a
    assert isinstance(obj2.inner.dep, HashValueFuru)


def test_from_dict_ignores_non_init_dataclass_fields() -> None:
    obj = DataclassWithNonInitField(value=5)
    data = furu.FuruSerializer.to_dict(obj)
    data["derived"] = 999
    restored = furu.FuruSerializer.from_dict(data)
    assert restored == obj
    assert restored.derived == 10


def test_from_dict_resolves_string_path_annotation() -> None:
    obj = DataclassWithStringPathAnnotation(p=Path("x/y"))
    data = furu.FuruSerializer.to_dict(obj)
    restored = furu.FuruSerializer.from_dict(data)
    assert restored == obj
    assert isinstance(restored.p, Path)


def test_from_dict_handles_unresolved_type_hints() -> None:
    obj = DataclassWithUnresolvedTypeHint(p=Path("x/y"), missing=cast(Any, 1))
    data = furu.FuruSerializer.to_dict(obj)
    restored = furu.FuruSerializer.from_dict(data)
    assert restored == obj
    assert isinstance(restored.p, Path)


def test_from_dict_rejects_incompatible_dataclass() -> None:
    bad_marker = furu.FuruSerializer.get_classname(RequiresTwoFields(1, 2))
    with pytest.raises(TypeError):
        furu.FuruSerializer.from_dict({"__class__": bad_marker, "first": 7})


def test_from_dict_rejects_unexpected_dataclass_fields_in_strict_mode() -> None:
    marker = furu.FuruSerializer.get_classname(RequiresTwoFields(1, 2))
    with pytest.raises(TypeError):
        furu.FuruSerializer.from_dict(
            {
                "__class__": marker,
                "first": 7,
                "second": 8,
                "extra": 9,
            }
        )


def test_from_dict_returns_dict_for_incompatible_dataclass_in_non_strict_mode() -> None:
    bad_marker = furu.FuruSerializer.get_classname(RequiresTwoFields(1, 2))
    restored = furu.FuruSerializer.from_dict(
        {"__class__": bad_marker, "first": 7},
        strict=False,
    )
    assert restored == {"first": 7}
    assert restored.first == 7


def test_from_dict_returns_dict_for_unexpected_dataclass_fields_in_non_strict_mode() -> (
    None
):
    marker = furu.FuruSerializer.get_classname(RequiresTwoFields(1, 2))
    restored = furu.FuruSerializer.from_dict(
        {
            "__class__": marker,
            "first": 7,
            "second": 8,
            "extra": 9,
        },
        strict=False,
    )
    assert restored == {"first": 7, "second": 8, "extra": 9}
    assert restored.extra == 9


def test_from_dict_non_strict_still_falls_back_on_signature_mismatch() -> None:
    marker = furu.FuruSerializer.get_classname(RequiresTwoArgs(1, 2))
    restored = furu.FuruSerializer.from_dict(
        {
            "__class__": marker,
            "first": 7,
        },
        strict=False,
    )
    assert restored == {"first": 7}
    assert restored.first == 7


def test_from_dict_non_strict_falls_back_on_custom_unexpected_keyword() -> None:
    marker = furu.FuruSerializer.get_classname(RequiresTwoArgs(1, 2))
    restored = furu.FuruSerializer.from_dict(
        {
            "__class__": marker,
            "first": 7,
            "second": 8,
            "extra": 9,
        },
        strict=False,
    )
    assert restored == {"first": 7, "second": 8, "extra": 9}


def test_from_dict_non_strict_falls_back_on_missing_keyword_only_arg() -> None:
    marker = furu.FuruSerializer.get_classname(RequiresKeywordOnlyArg(1, second=2))
    restored = furu.FuruSerializer.from_dict(
        {
            "__class__": marker,
            "first": 7,
        },
        strict=False,
    )
    assert restored == {"first": 7}


def test_from_dict_non_strict_reraises_non_schema_type_errors() -> None:
    marker = furu.FuruSerializer.get_classname(DataclassWithTypeErrorInPostInit(1))
    with pytest.raises(TypeError, match="post-init type error"):
        furu.FuruSerializer.from_dict(
            {
                "__class__": marker,
                "value": -1,
            },
            strict=False,
        )


def test_from_dict_returns_dict_for_missing_parent_module_path_in_non_strict_mode() -> (
    None
):
    payload = {"__class__": "no.such.module.Class", "value": 1}
    restored = furu.FuruSerializer.from_dict(payload, strict=False)
    assert restored == {"value": 1}
    assert restored.value == 1


def test_from_dict_rejects_unknown_class() -> None:
    payload = {"__class__": "test_serializer.DoesNotExist", "value": 1}
    with pytest.raises(AttributeError):
        furu.FuruSerializer.from_dict(payload)


def test_from_dict_returns_dict_for_unknown_class_in_non_strict_mode() -> None:
    payload = {"__class__": "test_serializer.DoesNotExist", "value": 1}
    restored = furu.FuruSerializer.from_dict(payload, strict=False)
    assert restored == {"value": 1}
    assert restored.value == 1


def test_from_dict_attr_access_prefers_payload_keys_over_dict_methods() -> None:
    payload = {
        "__class__": "test_serializer.DoesNotExist",
        "items": "value",
        "keys": "name",
    }
    restored = furu.FuruSerializer.from_dict(payload, strict=False)
    assert restored.items == "value"
    assert restored.keys == "name"


def test_from_dict_loads_nested_objects_if_possible() -> None:
    outer_marker = furu.FuruSerializer.get_classname(
        OuterWithNestedDataclass(inner=RequiresTwoFields(1, 2))
    )
    inner_marker = furu.FuruSerializer.get_classname(RequiresTwoFields(1, 2))
    payload = {
        "__class__": outer_marker,
        "inner": {
            "__class__": inner_marker,
            "first": 5,
        },
    }
    restored = furu.FuruSerializer.from_dict(payload, strict=False)
    assert isinstance(restored, OuterWithNestedDataclass)
    assert restored.inner == {"first": 5}
    assert restored.inner.first == 5


def test_from_dict_loads_nested_unknown_objects_with_attribute_access() -> None:
    outer_marker = furu.FuruSerializer.get_classname(
        OuterWithNestedDataclass(inner=RequiresTwoFields(1, 2))
    )
    restored = furu.FuruSerializer.from_dict(
        {
            "__class__": outer_marker,
            "inner": {
                "first": 5,
                "second": 6,
                "child": {
                    "__class__": "test_serializer.DoesNotExist",
                    "age": 8,
                },
            },
        },
        strict=False,
    )
    assert restored.inner.first == 5
    assert restored.inner.child == {"age": 8}
    assert restored.inner.child.age == 8
