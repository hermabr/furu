from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, ClassVar, cast

import pytest
from pydantic import BaseModel, ConfigDict

import furu
from furu import Furu
from furu._storage_layout import result_dir_in, result_manifest_path_in
from furu._declared_types import child_declared_type
from furu.result import (
    LazyResult,
    _save_result_bundle,
    load_result_bundle,
)
from furu.result.codec import (
    NumpyNpyCodec,
    PolarsParquetCodec,
    ResultCodec,
    ResultRegistry,
)

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")

_CHILD_DECLARED_TYPE_NAMESPACE: dict[str, object] = {
    "Annotated": Annotated,
    "Any": Any,
    "Ellipsis": Ellipsis,
    "LazyResult": LazyResult,
    "NumpyNpyCodec": NumpyNpyCodec,
    "Path": Path,
    "ResultCodec": ResultCodec,
    "bool": bool,
    "dict": dict,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "list": list,
    "object": object,
    "set": set,
    "str": str,
    "tuple": tuple,
}
_CHILD_DECLARED_TYPE_GLOBALS = {"__builtins__": {"__import__": __import__}}


class JsonResult(Furu[dict[str, object]]):
    create_calls: ClassVar[list[int]] = []

    def create(self) -> dict[str, object]:
        type(self).create_calls.append(1)
        return {
            "metrics": {"loss": 0.12, "ok": True},
            "items": [1, 2, None, "x"],
        }


def test_json_only_bundle_round_trips() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    expected = {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }

    result = obj.create()
    assert result == expected

    assert result_manifest_path_in(obj._base_dir).exists()
    assert not (result_dir_in(obj._base_dir) / "artifacts").exists()

    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert "format" not in manifest
    assert "root" not in manifest
    assert manifest == expected


def test_json_only_cache_hit_does_not_recompute() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    expected = {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }

    assert obj.create() == expected
    assert obj.create() == expected
    assert len(JsonResult.create_calls) == 1


def test_status_is_completed_after_first_run() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    assert obj.status() == "missing"
    obj.create()
    assert obj.status() == "completed"


def test_load_existing_returns_persisted_result() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    obj.create()

    assert obj.load_existing() == {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }


class ScalarResult(Furu[int]):
    def create(self) -> int:
        return 5


def test_scalar_root_manifest_is_just_the_value() -> None:
    obj = ScalarResult()

    assert obj.create() == 5
    text = result_manifest_path_in(obj._base_dir).read_text()
    assert json.loads(text) == 5


class PathResult(Furu[dict[str, Path]]):
    def create(self) -> dict[str, Path]:
        return {
            "relative": Path("outputs/model.bin"),
            "absolute": Path("/tmp/furu/model.bin"),
        }


def test_path_values_round_trip() -> None:
    obj = PathResult()

    result = obj.create()

    assert result == {
        "relative": Path("outputs/model.bin"),
        "absolute": Path("/tmp/furu/model.bin"),
    }

    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest == {
        "relative": {
            "$furu": {
                "|kind": "path",
                "value": "outputs/model.bin",
            }
        },
        "absolute": {
            "$furu": {
                "|kind": "path",
                "value": "/tmp/furu/model.bin",
            }
        },
    }


def test_path_root_value_round_trips(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = Path("outputs/model.bin")

    _save_result_bundle(value, bundle_dir, registry=ResultRegistry.default())

    assert load_result_bundle(bundle_dir) == value


def test_tuple_set_and_frozenset_round_trip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = {
        "tuple": (1, "x", Path("model.bin")),
        "set": {3, 1, 2},
        "frozenset": frozenset({"b", "a"}),
    }

    _save_result_bundle(value, bundle_dir, registry=ResultRegistry.default())

    assert load_result_bundle(bundle_dir) == value
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["tuple"] == {
        "$furu": {
            "|kind": "tuple",
            "items": [
                1,
                "x",
                {"$furu": {"|kind": "path", "value": "model.bin"}},
            ],
        }
    }
    assert manifest["set"]["$furu"] == {"|kind": "set", "items": [1, 2, 3]}
    assert manifest["frozenset"]["$furu"] == {
        "|kind": "frozenset",
        "items": ["a", "b"],
    }


def test_tuple_root_value_uses_furu_wrapper(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = (1, 2, 3)

    _save_result_bundle(value, bundle_dir, registry=ResultRegistry.default())

    assert load_result_bundle(bundle_dir) == value
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest == {"$furu": {"|kind": "tuple", "items": [1, 2, 3]}}


class NonFiniteFloatResult(Furu[dict[str, float]]):
    def create(self) -> dict[str, float]:
        return {
            "nan": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
        }


def test_non_finite_floats_round_trip() -> None:
    obj = NonFiniteFloatResult()
    result = obj.create()

    assert math.isnan(result["nan"])
    assert result["pos_inf"] == float("inf")
    assert result["neg_inf"] == float("-inf")

    text = result_manifest_path_in(obj._base_dir).read_text()
    # Python's standard library writes these as JSON extensions.
    assert "NaN" in text
    assert "Infinity" in text
    assert "-Infinity" in text


class _CustomTensor:
    pass


class _CountingValue:
    def __init__(self, value: int) -> None:
        self.value = value


class _CountingCodec(ResultCodec[_CountingValue]):
    auto_register: ClassVar[bool] = False
    dump_calls: ClassVar[int] = 0
    load_calls: ClassVar[int] = 0

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _CountingValue)

    @classmethod
    def dump(
        cls,
        value: _CountingValue,
        *,
        artifact_dir: Path,
    ) -> None:
        cls.dump_calls += 1
        artifact_dir.joinpath("value.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, *, artifact_dir: Path) -> _CountingValue:
        cls.load_calls += 1
        return _CountingValue(
            int(artifact_dir.joinpath("value.txt").read_text(encoding="utf-8"))
        )


class _OtherCountingCodec(ResultCodec[_CountingValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _CountingValue)

    @classmethod
    def dump(
        cls,
        value: _CountingValue,
        *,
        artifact_dir: Path,
    ) -> None:
        artifact_dir.joinpath("other.txt").write_text("x", encoding="utf-8")

    @classmethod
    def load(cls, *, artifact_dir: Path) -> _CountingValue:
        artifact_dir.joinpath("other.txt").read_text(encoding="utf-8")
        return _CountingValue(0)


class _CustomNumpyCodec(ResultCodec[Any]):
    auto_register: ClassVar[bool] = False
    file_name: ClassVar[str] = "custom.npy"

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, np.ndarray)

    @classmethod
    def dump(
        cls,
        value: Any,
        *,
        artifact_dir: Path,
    ) -> None:
        np.save(artifact_dir / cls.file_name, value, allow_pickle=False)

    @classmethod
    def load(cls, *, artifact_dir: Path) -> Any:
        return np.load(artifact_dir / cls.file_name, allow_pickle=False)


class _RegistryNumpyCodec(_CustomNumpyCodec):
    auto_register: ClassVar[bool] = False
    file_name: ClassVar[str] = "registry.npy"


class _AutoRegisteredValue:
    def __init__(self, value: int) -> None:
        self.value = value


class _AutoRegisteredValueCodec(ResultCodec[_AutoRegisteredValue]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredValue)

    @classmethod
    def dump(
        cls,
        value: _AutoRegisteredValue,
        *,
        artifact_dir: Path,
    ) -> None:
        artifact_dir.joinpath("auto.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, *, artifact_dir: Path) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(
            int(artifact_dir.joinpath("auto.txt").read_text(encoding="utf-8"))
        )


class _CoreRegistryAutoValueCodec(ResultCodec[_AutoRegisteredValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredValue)

    @classmethod
    def dump(
        cls,
        value: _AutoRegisteredValue,
        *,
        artifact_dir: Path,
    ) -> None:
        artifact_dir.joinpath("registry.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, *, artifact_dir: Path) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(
            int(artifact_dir.joinpath("registry.txt").read_text(encoding="utf-8"))
        )


class _AutoRegisteredArray(np.ndarray):
    pass


class _AutoRegisteredArrayCodec(ResultCodec[Any]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredArray)

    @classmethod
    def dump(
        cls,
        value: Any,
        *,
        artifact_dir: Path,
    ) -> None:
        np.save(
            artifact_dir / "auto.npy",
            value.view(np.ndarray),
            allow_pickle=False,
        )

    @classmethod
    def load(cls, *, artifact_dir: Path) -> Any:
        return np.load(artifact_dir / "auto.npy", allow_pickle=False)


class _OptOutRegisteredValue:
    pass


class _OptOutRegisteredValueCodec(ResultCodec[_OptOutRegisteredValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _OptOutRegisteredValue)

    @classmethod
    def dump(
        cls,
        value: _OptOutRegisteredValue,
        *,
        artifact_dir: Path,
    ) -> None:
        artifact_dir.joinpath("manual.txt").write_text("", encoding="utf-8")

    @classmethod
    def load(cls, *, artifact_dir: Path) -> _OptOutRegisteredValue:
        artifact_dir.joinpath("manual.txt").read_text(encoding="utf-8")
        return _OptOutRegisteredValue()


class RegistryAutoRegisteredValueResult(Furu[_AutoRegisteredValue]):
    @property
    def result_registry(self) -> ResultRegistry:
        return super().result_registry.with_codec(_CoreRegistryAutoValueCodec)

    def create(self) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(10)


class AnnotatedAutoRegisteredValueResult(
    Furu[Annotated[_AutoRegisteredValue, _CoreRegistryAutoValueCodec]]
):
    def create(self) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(11)


def test_codec_id_is_derived_from_class_identity() -> None:
    assert _CountingCodec._codec_id() == (
        f"{_CountingCodec.__module__}.{_CountingCodec.__qualname__}"
    )


def test_result_registry_with_codec_is_functional() -> None:
    first = ResultRegistry.default().with_codec(_CountingCodec)
    second = first.with_codec(_OtherCountingCodec)

    assert ResultRegistry.default().find_codec(_CountingValue(1)) is None
    assert first.find_codec(_CountingValue(1)) is _CountingCodec
    with pytest.raises(TypeError, match="result registry matched multiple codecs"):
        second.find_codec(_CountingValue(1))


def test_user_defined_codec_is_auto_registered(tmp_path: Path) -> None:
    assert (
        ResultRegistry.default().find_codec(_AutoRegisteredValue(1))
        is _AutoRegisteredValueCodec
    )

    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        _AutoRegisteredValue(3),
        bundle_dir,
        registry=ResultRegistry.default(),
    )

    artifact_dir = bundle_dir / "artifacts" / "root"
    assert (artifact_dir / "auto.txt").exists()
    assert not (artifact_dir / "registry.txt").exists()
    loaded = load_result_bundle(bundle_dir)
    assert isinstance(loaded, _AutoRegisteredValue)
    assert loaded.value == 3


def test_auto_register_false_opts_out_of_default_registry(tmp_path: Path) -> None:
    assert ResultRegistry.default().find_codec(_OptOutRegisteredValue()) is None

    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        _OptOutRegisteredValue(),
        bundle_dir,
        registry=ResultRegistry.default().with_codec(_OptOutRegisteredValueCodec),
    )

    assert (bundle_dir / "artifacts" / "root" / "manual.txt").exists()
    loaded = load_result_bundle(bundle_dir)
    assert isinstance(loaded, _OptOutRegisteredValue)


def test_auto_registered_codec_takes_priority_over_builtin_codec(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    value = np.arange(3, dtype=np.int64).view(_AutoRegisteredArray)

    _save_result_bundle(value, bundle_dir, registry=ResultRegistry.default())

    artifact_dir = bundle_dir / "artifacts" / "root"
    assert (artifact_dir / "auto.npy").exists()
    assert not (artifact_dir / "data.npy").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["$furu"]["codec"] == _AutoRegisteredArrayCodec._codec_id()


def test_task_result_registry_takes_priority_over_auto_registered_codec() -> None:
    obj = RegistryAutoRegisteredValueResult()

    loaded = obj.create()

    assert isinstance(loaded, _AutoRegisteredValue)
    assert loaded.value == 10
    artifact_dir = result_dir_in(obj._base_dir) / "artifacts" / "root"
    assert (artifact_dir / "registry.txt").exists()
    assert not (artifact_dir / "auto.txt").exists()
    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest["$furu"]["codec"] == _CoreRegistryAutoValueCodec._codec_id()


def test_annotated_codec_takes_priority_over_auto_registered_codec() -> None:
    obj = AnnotatedAutoRegisteredValueResult()

    loaded = obj.create()

    assert isinstance(loaded, _AutoRegisteredValue)
    assert loaded.value == 11
    artifact_dir = result_dir_in(obj._base_dir) / "artifacts" / "root"
    assert (artifact_dir / "registry.txt").exists()
    assert not (artifact_dir / "auto.txt").exists()
    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest["$furu"]["codec"] == _CoreRegistryAutoValueCodec._codec_id()


def test_codec_defined_after_default_cache_is_auto_registered() -> None:
    class LateAutoRegisteredValue:
        pass

    assert ResultRegistry.default().find_codec(LateAutoRegisteredValue()) is None

    class LateAutoRegisteredCodec(ResultCodec[LateAutoRegisteredValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, LateAutoRegisteredValue)

        @classmethod
        def dump(
            cls,
            value: LateAutoRegisteredValue,
            *,
            artifact_dir: Path,
        ) -> None:
            artifact_dir.joinpath("late.txt").write_text("", encoding="utf-8")

        @classmethod
        def load(cls, *, artifact_dir: Path) -> LateAutoRegisteredValue:
            artifact_dir.joinpath("late.txt").read_text(encoding="utf-8")
            return LateAutoRegisteredValue()

    assert (
        ResultRegistry.default().find_codec(LateAutoRegisteredValue())
        is LateAutoRegisteredCodec
    )


def test_auto_registered_codecs_must_not_be_ambiguous() -> None:
    class AutoAmbiguousValue:
        pass

    class FirstAutoAmbiguousCodec(ResultCodec[AutoAmbiguousValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, AutoAmbiguousValue)

        @classmethod
        def dump(
            cls,
            value: AutoAmbiguousValue,
            *,
            artifact_dir: Path,
        ) -> None:
            artifact_dir.joinpath("first.txt").write_text("", encoding="utf-8")

        @classmethod
        def load(cls, *, artifact_dir: Path) -> AutoAmbiguousValue:
            artifact_dir.joinpath("first.txt").read_text(encoding="utf-8")
            return AutoAmbiguousValue()

    class SecondAutoAmbiguousCodec(ResultCodec[AutoAmbiguousValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, AutoAmbiguousValue)

        @classmethod
        def dump(
            cls,
            value: AutoAmbiguousValue,
            *,
            artifact_dir: Path,
        ) -> None:
            artifact_dir.joinpath("second.txt").write_text("", encoding="utf-8")

        @classmethod
        def load(cls, *, artifact_dir: Path) -> AutoAmbiguousValue:
            artifact_dir.joinpath("second.txt").read_text(encoding="utf-8")
            return AutoAmbiguousValue()

    with pytest.raises(TypeError) as exc_info:
        ResultRegistry.default().find_codec(AutoAmbiguousValue())

    message = str(exc_info.value)
    assert "auto-registered codec registry matched multiple codecs" in message
    assert "FirstAutoAmbiguousCodec" in message
    assert "SecondAutoAmbiguousCodec" in message


@pytest.mark.parametrize(
    ("declared_type_expr", "key", "expected_type_expr"),
    [
        ("list[int]", 0, "int"),
        ("list[int]", 12, "int"),
        ("list[Annotated[int, NumpyNpyCodec]]", 0, "Annotated[int, NumpyNpyCodec]"),
        ("Annotated[list[str], NumpyNpyCodec]", 0, "str"),
        ("dict[str, float]", "loss", "float"),
        ("dict[int, Path]", 7, "Path"),
        (
            "dict[str, Annotated[Any, NumpyNpyCodec]]",
            "weights",
            "Annotated[Any, NumpyNpyCodec]",
        ),
        ("Annotated[dict[str, int], NumpyNpyCodec]", "value", "int"),
        ("tuple[int, ...]", 0, "int"),
        ("tuple[int, ...]", 99, "int"),
        (
            "tuple[Annotated[Any, NumpyNpyCodec], ...]",
            3,
            "Annotated[Any, NumpyNpyCodec]",
        ),
        ("Annotated[tuple[str, ...], NumpyNpyCodec]", 3, "str"),
        ("tuple[int, str, Path]", 0, "int"),
        ("tuple[int, str, Path]", 1, "str"),
        ("tuple[int, str, Path]", 2, "Path"),
        (
            "tuple[int, Annotated[Any, NumpyNpyCodec], Path]",
            1,
            "Annotated[Any, NumpyNpyCodec]",
        ),
        ("Annotated[tuple[int, str], NumpyNpyCodec]", 1, "str"),
        ("tuple[int, str]", 2, "Any"),
        ("tuple[int, str]", "not-an-index", "Any"),
        ("set[int]", 0, "int"),
        ("set[Annotated[Any, NumpyNpyCodec]]", 0, "Annotated[Any, NumpyNpyCodec]"),
        ("Annotated[set[str], NumpyNpyCodec]", 0, "str"),
        ("frozenset[int]", 0, "int"),
        (
            "frozenset[Annotated[Any, NumpyNpyCodec]]",
            0,
            "Annotated[Any, NumpyNpyCodec]",
        ),
        ("Annotated[frozenset[str], NumpyNpyCodec]", 0, "str"),
        ("object", 0, "Any"),
        ("Any", 0, "Any"),
    ],
)
def test_child_declared_type_descends_supported_container_annotations(
    declared_type_expr: str,
    key: object,
    expected_type_expr: str,
) -> None:
    declared_type = eval(
        declared_type_expr, _CHILD_DECLARED_TYPE_GLOBALS, _CHILD_DECLARED_TYPE_NAMESPACE
    )
    expected_type = eval(
        expected_type_expr, _CHILD_DECLARED_TYPE_GLOBALS, _CHILD_DECLARED_TYPE_NAMESPACE
    )

    assert child_declared_type(declared_type, key) == expected_type


@dataclass(frozen=True)
class AnnotatedArrayOutput:
    weights: Annotated[Any, NumpyNpyCodec]


class AnnotatedArrayResult(Furu[AnnotatedArrayOutput]):
    def create(self) -> AnnotatedArrayOutput:
        return AnnotatedArrayOutput(weights=np.arange(3, dtype=np.int64))


class GenericAnnotatedArrayBase[T](Furu[T]):
    def create(self) -> T:
        return np.arange(3, dtype=np.int64)


class GenericAnnotatedArrayResult(
    GenericAnnotatedArrayBase[Annotated[Any, NumpyNpyCodec]]
):
    pass


class StrictAnnotatedArrayOutput(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    weights: Annotated[np.ndarray[Any, Any], NumpyNpyCodec]


class StrictAnnotatedArrayResult(Furu[StrictAnnotatedArrayOutput]):
    def create(self) -> StrictAnnotatedArrayOutput:
        return StrictAnnotatedArrayOutput(weights=np.arange(3, dtype=np.int64))


def test_annotated_codec_selects_external_artifact() -> None:
    obj = AnnotatedArrayResult()
    loaded = obj.create()

    assert isinstance(loaded, AnnotatedArrayOutput)
    assert np.array_equal(loaded.weights, np.arange(3, dtype=np.int64))
    assert (
        result_dir_in(obj._base_dir) / "artifacts" / "weights" / "data.npy"
    ).exists()


def test_generic_furu_base_with_annotated_result_codec_is_rejected() -> None:
    obj = GenericAnnotatedArrayResult()

    with pytest.raises(TypeError, match="concrete result type directly as Furu"):
        obj.create()


def test_strict_pydantic_annotated_codec_selects_external_artifact() -> None:
    obj = StrictAnnotatedArrayResult()
    loaded = obj.create()

    assert isinstance(loaded, StrictAnnotatedArrayOutput)
    assert np.array_equal(loaded.weights, np.arange(3, dtype=np.int64))
    assert (
        result_dir_in(obj._base_dir) / "artifacts" / "weights" / "data.npy"
    ).exists()

    loaded_again = obj.create()
    assert isinstance(loaded_again, StrictAnnotatedArrayOutput)
    assert np.array_equal(loaded_again.weights, np.arange(3, dtype=np.int64))


@dataclass(frozen=True)
class SaveAsOutput:
    weights: Any


class SaveAsArrayResult(Furu[SaveAsOutput]):
    def create(self) -> SaveAsOutput:
        return SaveAsOutput(weights=furu.save_as(np.arange(4), codec=NumpyNpyCodec))


@dataclass(frozen=True)
class LazySaveAsOutput:
    weights: LazyResult[Any]


class LazySaveAsArrayResult(Furu[LazySaveAsOutput]):
    def create(self) -> LazySaveAsOutput:
        return LazySaveAsOutput(
            weights=LazyResult(furu.save_as(np.arange(4), codec=NumpyNpyCodec))
        )


def test_save_as_selects_codec_and_does_not_leak_wrapper() -> None:
    obj = SaveAsArrayResult()
    loaded = obj.create()

    assert isinstance(loaded, SaveAsOutput)
    assert np.array_equal(loaded.weights, np.arange(4))
    assert type(loaded.weights).__name__ != "_SaveAs"

    loaded_again = obj.create()
    assert isinstance(loaded_again, SaveAsOutput)
    assert np.array_equal(loaded_again.weights, np.arange(4))


def test_save_as_inside_lazy_result_does_not_leak_wrapper() -> None:
    obj = LazySaveAsArrayResult()
    loaded = obj.create()

    assert isinstance(loaded, LazySaveAsOutput)
    assert isinstance(loaded.weights, LazyResult)
    assert loaded.weights.is_loaded
    assert np.array_equal(loaded.weights.load(), np.arange(4))
    assert type(loaded.weights.load()).__name__ != "_SaveAs"

    loaded_again = obj.create()
    assert isinstance(loaded_again, LazySaveAsOutput)
    assert isinstance(loaded_again.weights, LazyResult)
    assert not loaded_again.weights.is_loaded
    assert np.array_equal(loaded_again.weights.load(), np.arange(4))
    assert type(loaded_again.weights.load()).__name__ != "_SaveAs"


@dataclass(frozen=True)
class ConflictingSaveAsOutput:
    weights: Annotated[Any, NumpyNpyCodec]


class ConflictingSaveAsResult(Furu[ConflictingSaveAsOutput]):
    def create(self) -> ConflictingSaveAsOutput:
        return ConflictingSaveAsOutput(
            weights=furu.save_as(np.arange(4), codec=_OtherCountingCodec)
        )


def test_save_as_conflicts_with_annotated_codec() -> None:
    with pytest.raises(TypeError, match="Conflicting codecs"):
        ConflictingSaveAsResult().create()


class RegistryCountingResult(Furu[_CountingValue]):
    @property
    def result_registry(self) -> ResultRegistry:
        return super().result_registry.with_codec(_CountingCodec)

    def create(self) -> _CountingValue:
        return _CountingValue(8)


class AmbiguousRegistryCountingResult(Furu[_CountingValue]):
    @property
    def result_registry(self) -> ResultRegistry:
        return (
            super()
            .result_registry.with_codec(_CountingCodec)
            .with_codec(_OtherCountingCodec)
        )

    def create(self) -> _CountingValue:
        return _CountingValue(8)


def test_task_result_registry_is_used_for_save_inference_only() -> None:
    _CountingCodec.dump_calls = 0
    _CountingCodec.load_calls = 0
    obj = RegistryCountingResult()

    loaded = obj.create()
    assert isinstance(loaded, _CountingValue)
    assert loaded.value == 8
    assert _CountingCodec.dump_calls == 1
    assert _CountingCodec.load_calls == 0

    loaded_again = obj.create()
    assert isinstance(loaded_again, _CountingValue)
    assert loaded_again.value == 8
    assert _CountingCodec.load_calls == 1


def test_task_result_registry_must_not_be_ambiguous() -> None:
    with pytest.raises(TypeError, match="result registry matched multiple codecs"):
        AmbiguousRegistryCountingResult().create()


class UnsupportedRootResult(Furu[object]):
    def create(self) -> object:
        return _CustomTensor()


def test_unsupported_custom_object_fails_with_root_path(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle(
            _CustomTensor(), bundle_dir, registry=ResultRegistry.default()
        )
    msg = str(exc_info.value)
    assert "<root>" in msg
    assert "_CustomTensor" in msg


def test_unsupported_nested_path_includes_padded_index(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    layers = [{} for _ in range(10)]
    layers[3] = {"weights": _CustomTensor()}

    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle(
            {"layers": layers}, bundle_dir, registry=ResultRegistry.default()
        )
    assert "layers/03/weights" in str(exc_info.value)


def test_reserved_furu_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="reserved"):
        _save_result_bundle(
            {"$furu": "user data"}, bundle_dir, registry=ResultRegistry.default()
        )


def test_non_string_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="must be strings"):
        _save_result_bundle({1: "x"}, bundle_dir, registry=ResultRegistry.default())


def test_unsafe_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle(
            {"bad/key": "x"}, bundle_dir, registry=ResultRegistry.default()
        )
    assert "artifact path segment" in str(exc_info.value)


def test_empty_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle({"": "x"}, bundle_dir, registry=ResultRegistry.default())
    assert "artifact path segment" in str(exc_info.value)


def test_dotdot_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="artifact path segment"):
        _save_result_bundle({"..": "x"}, bundle_dir, registry=ResultRegistry.default())


@dataclass(frozen=True)
class TrainOutput:
    metrics: dict[str, float]
    values: list[int]


class DataclassResult(Furu[TrainOutput]):
    def create(self) -> TrainOutput:
        return TrainOutput(metrics={"loss": 0.12}, values=[1, 2, 3])


def test_dataclass_round_trip() -> None:
    obj = DataclassResult()
    loaded = obj.create()
    assert isinstance(loaded, TrainOutput)
    assert loaded == TrainOutput(metrics={"loss": 0.12}, values=[1, 2, 3])

    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest["$furu"]["|kind"] == "dataclass"
    assert manifest["$furu"]["|type"] == "test_result.TrainOutput"
    assert manifest["$furu"]["|fields"] == {
        "metrics": {"loss": 0.12},
        "values": [1, 2, 3],
    }


@dataclass(frozen=True)
class DataclassWithPostInit:
    value: int
    doubled: int = field(init=False)

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("value must be non-negative")
        object.__setattr__(self, "doubled", self.value * 2)


def test_dataclass_load_uses_constructor(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        DataclassWithPostInit(value=3), bundle_dir, registry=ResultRegistry.default()
    )

    loaded = load_result_bundle(bundle_dir)

    assert loaded == DataclassWithPostInit(value=3)


def test_dataclass_load_reports_constructor_error_with_path(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        {"result": DataclassWithPostInit(value=3)},
        bundle_dir,
        registry=ResultRegistry.default(),
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["result"]["$furu"]["|fields"]["value"] = -1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_result_bundle(bundle_dir)

    message = str(exc_info.value)
    assert "Cannot load dataclass" in message
    assert "at result" in message
    assert "value must be non-negative" in message


def test_dataclass_load_reports_missing_and_extra_fields_with_path(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        {"result": TrainOutput(metrics={"loss": 0.12}, values=[1])},
        bundle_dir,
        registry=ResultRegistry.default(),
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fields = manifest["result"]["$furu"]["|fields"]
    fields["renamed_values"] = fields.pop("values")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_result_bundle(bundle_dir)

    message = str(exc_info.value)
    assert "at result" in message
    assert "missing fields: values" in message
    assert "extra fields: renamed_values" in message


@dataclass(frozen=True)
class NestedOuter:
    inner: "NestedInner"
    label: str


@dataclass(frozen=True)
class NestedInner:
    value: int


class NestedDataclassResult(Furu[NestedOuter]):
    def create(self) -> NestedOuter:
        return NestedOuter(inner=NestedInner(value=42), label="root")


def test_nested_dataclass_round_trip() -> None:
    obj = NestedDataclassResult()
    loaded = obj.create()
    assert isinstance(loaded, NestedOuter)
    assert isinstance(loaded.inner, NestedInner)
    assert loaded == NestedOuter(inner=NestedInner(value=42), label="root")


class TrainOutputModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    metrics: dict[str, float]
    values: list[int]


class PydanticResult(Furu[TrainOutputModel]):
    def create(self) -> TrainOutputModel:
        return TrainOutputModel(metrics={"loss": 0.12}, values=[1, 2, 3])


def test_pydantic_round_trip() -> None:
    obj = PydanticResult()
    loaded = obj.create()
    assert isinstance(loaded, TrainOutputModel)
    assert loaded.metrics == {"loss": 0.12}
    assert loaded.values == [1, 2, 3]


class ValidatedTrainOutputModel(BaseModel):
    value: int


def test_pydantic_load_uses_model_validate(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        ValidatedTrainOutputModel(value=1),
        bundle_dir,
        registry=ResultRegistry.default(),
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["$furu"]["|fields"]["value"] = "not an int"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_result_bundle(bundle_dir)

    message = str(exc_info.value)
    assert "Cannot load pydantic model" in message
    assert "at <root>" in message
    assert "value" in message


def test_pydantic_load_reports_missing_and_extra_fields_with_path(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        {"models": [TrainOutputModel(metrics={}, values=[])]},
        bundle_dir,
        registry=ResultRegistry.default(),
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fields = manifest["models"][0]["$furu"]["|fields"]
    fields["renamed_values"] = fields.pop("values")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_result_bundle(bundle_dir)

    message = str(exc_info.value)
    assert "at models/0" in message
    assert "missing fields: values" in message
    assert "extra fields: renamed_values" in message


class NestedPydanticOuter(BaseModel):
    metrics: dict[str, float]
    items: list[dict[str, int]]


class NestedPydanticResult(Furu[NestedPydanticOuter]):
    def create(self) -> NestedPydanticOuter:
        return NestedPydanticOuter(
            metrics={"loss": 0.12, "accuracy": 0.94},
            items=[{"v": 1}, {"v": 2}, {"v": 3}],
        )


def test_pydantic_with_nested_structures_round_trips() -> None:
    obj = NestedPydanticResult()
    loaded = obj.create()
    assert isinstance(loaded, NestedPydanticOuter)
    assert loaded.metrics == {"loss": 0.12, "accuracy": 0.94}
    assert loaded.items == [{"v": 1}, {"v": 2}, {"v": 3}]


class NumpyResult(Furu[dict[str, object]]):
    def create(self) -> dict[str, object]:
        return {"weights": np.arange(10, dtype=np.float32)}


class RegistryNumpyResult(Furu[Any]):
    @property
    def result_registry(self) -> ResultRegistry:
        return super().result_registry.with_codec(_RegistryNumpyCodec)

    def create(self) -> Any:
        return np.arange(10, dtype=np.float32)


class _MemmapNumpyNpyCodec(NumpyNpyCodec):
    auto_register: ClassVar[bool] = False
    load_after_dump: ClassVar[bool] = True

    @classmethod
    def load(cls, *, artifact_dir: Path) -> np.ndarray[Any, Any]:
        return np.load(artifact_dir / "data.npy", allow_pickle=False, mmap_mode="r")


class MemmapNumpyResult(Furu[Annotated[Any, _MemmapNumpyNpyCodec]]):
    def create(self) -> np.ndarray[Any, Any]:
        return np.arange(10, dtype=np.float32)


def test_numpy_array_round_trips() -> None:
    obj = NumpyResult()
    loaded = obj.create()

    assert (
        result_dir_in(obj._base_dir) / "artifacts" / "weights" / "data.npy"
    ).exists()
    assert isinstance(loaded, dict)
    weights = cast(Any, loaded["weights"])
    assert weights.dtype == np.float32
    assert np.array_equal(weights, np.arange(10, dtype=np.float32))

    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest["weights"]["$furu"]["|kind"] == "external"
    assert manifest["weights"]["$furu"]["codec"] == (
        f"{NumpyNpyCodec.__module__}.{NumpyNpyCodec.__qualname__}"
    )
    assert manifest["weights"]["$furu"]["path"] == "artifacts/weights"


def test_result_registry_takes_priority_over_builtin_codec() -> None:
    obj = RegistryNumpyResult()

    loaded = obj.create()
    loaded_again = obj.create()

    assert np.array_equal(loaded, np.arange(10, dtype=np.float32))
    assert np.array_equal(loaded_again, np.arange(10, dtype=np.float32))
    artifact_dir = result_dir_in(obj._base_dir) / "artifacts" / "root"
    assert (artifact_dir / "registry.npy").exists()
    assert not (artifact_dir / "data.npy").exists()
    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest["$furu"]["codec"] == _RegistryNumpyCodec._codec_id()


def test_codec_can_force_load_after_dump_for_cache_miss_consistency() -> None:
    obj = MemmapNumpyResult()
    expected_file = result_dir_in(obj._base_dir) / "artifacts" / "root" / "data.npy"

    first = obj.create()
    second = obj.create()

    assert isinstance(first, np.memmap)
    assert isinstance(second, np.memmap)
    assert Path(first.filename).resolve() == expected_file.resolve()
    assert Path(second.filename).resolve() == expected_file.resolve()
    assert np.array_equal(first, np.arange(10, dtype=np.float32))
    assert np.array_equal(second, np.arange(10, dtype=np.float32))


class PolarsResult(Furu[dict[str, object]]):
    def create(self) -> dict[str, object]:
        return {"frame": pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})}


def test_polars_dataframe_round_trips() -> None:
    obj = PolarsResult()
    loaded = obj.create()

    assert (
        result_dir_in(obj._base_dir) / "artifacts" / "frame" / "data.parquet"
    ).exists()
    assert isinstance(loaded, dict)
    frame = loaded["frame"]
    assert isinstance(frame, pl.DataFrame)
    assert frame.equals(pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}))

    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest["frame"]["$furu"]["|kind"] == "external"
    assert manifest["frame"]["$furu"]["codec"] == (
        f"{PolarsParquetCodec.__module__}.{PolarsParquetCodec.__qualname__}"
    )
    assert manifest["frame"]["$furu"]["path"] == "artifacts/frame"


def test_numpy_object_dtype_is_rejected(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="allow_pickle=False"):
        _save_result_bundle(
            {"weights": np.array([object()], dtype=object)},
            bundle_dir,
            registry=ResultRegistry.default(),
        )


class NestedNumpyResult(Furu[dict[str, list[dict[str, object]]]]):
    def create(self) -> dict[str, list[dict[str, object]]]:
        return {
            "layers": [{"weights": np.arange(i, dtype=np.float32)} for i in range(10)]
        }


def test_nested_numpy_paths_use_list_length_padded_indexes() -> None:
    obj = NestedNumpyResult()
    loaded = obj.create()

    layers_dir = result_dir_in(obj._base_dir) / "artifacts" / "layers"
    for i in range(10):
        weights_file = layers_dir / f"{i:02d}" / "weights" / "data.npy"
        assert weights_file.exists(), f"missing {weights_file}"

    assert isinstance(loaded, dict)
    layers = loaded["layers"]
    assert len(layers) == 10
    for i, layer in enumerate(layers):
        assert isinstance(layer, dict)
        assert np.array_equal(layer["weights"], np.arange(i, dtype=np.float32))


def test_long_list_uses_three_digit_padding(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    layers = [{"weights": np.arange(0, dtype=np.float32)} for _ in range(100)]
    layers[3] = {"weights": np.arange(3, dtype=np.float32)}

    _save_result_bundle(
        {"layers": layers}, bundle_dir, registry=ResultRegistry.default()
    )

    expected = bundle_dir / "artifacts" / "layers" / "003" / "weights" / "data.npy"
    assert expected.exists()


def test_numpy_root_value_uses_root_artifact_dir(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        np.arange(5, dtype=np.int64), bundle_dir, registry=ResultRegistry.default()
    )

    assert (bundle_dir / "artifacts" / "root" / "data.npy").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["$furu"]["|kind"] == "external"
    assert manifest["$furu"]["path"] == "artifacts/root"

    loaded = load_result_bundle(bundle_dir)
    assert np.array_equal(loaded, np.arange(5, dtype=np.int64))


@dataclass(frozen=True)
class MixedOutput:
    metrics: dict[str, float]
    values: list[int]


class MixedResult(Furu[dict[str, object]]):
    def create(self) -> dict[str, object]:
        return {
            "result": MixedOutput(metrics={"loss": 0.5}, values=[1, 2]),
            "weights": np.arange(4, dtype=np.float32),
            "labels": ["cat", "dog"],
        }


def test_mixed_dataclass_external_and_json_round_trip() -> None:
    obj = MixedResult()
    loaded = obj.create()

    assert isinstance(loaded, dict)
    inner = loaded["result"]
    assert isinstance(inner, MixedOutput)
    assert inner == MixedOutput(metrics={"loss": 0.5}, values=[1, 2])
    assert np.array_equal(loaded["weights"], np.arange(4, dtype=np.float32))
    assert loaded["labels"] == ["cat", "dog"]


def test_private_save_result_bundle_refuses_existing_directory(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    with pytest.raises(FileExistsError):
        _save_result_bundle({"x": 1}, bundle_dir, registry=ResultRegistry.default())


def test_private_save_result_bundle_writes_manifest_last(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        {"weights": np.arange(2, dtype=np.float32)},
        bundle_dir,
        registry=ResultRegistry.default(),
    )

    # All three pieces should now be present.
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "artifacts" / "weights" / "data.npy").exists()


def test_load_result_bundle_rejects_artifacts_path_escape(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "artifacts").mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "$furu": {
                    "|kind": "external",
                    "codec": f"{NumpyNpyCodec.__module__}.{NumpyNpyCodec.__qualname__}",
                    "path": "../../../etc/passwd",
                }
            }
        )
    )

    with pytest.raises(ValueError, match="escapes"):
        load_result_bundle(bundle_dir)


def test_lazy_result_created_directly_is_loaded() -> None:
    value = _CountingValue(7)
    lazy = LazyResult(value)

    assert lazy.is_loaded
    assert repr(lazy) == "LazyResult(_CountingValue)"
    assert lazy.load() is value
    with pytest.raises(RuntimeError, match="only available after persistence"):
        lazy.path


def test_root_lazy_result_defers_cache_read_and_memoizes(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    registry = ResultRegistry.default().with_codec(_CountingCodec)
    _CountingCodec.dump_calls = 0
    _CountingCodec.load_calls = 0

    _save_result_bundle(LazyResult(_CountingValue(9)), bundle_dir, registry=registry)

    assert _CountingCodec.dump_calls == 1
    assert _CountingCodec.load_calls == 0
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest == {"$furu": {"|kind": "lazy", "path": "lazy/root"}}
    assert (bundle_dir / "lazy" / "root" / "manifest.json").exists()

    loaded = load_result_bundle(bundle_dir)

    assert isinstance(loaded, LazyResult)
    assert not loaded.is_loaded
    assert loaded.path == bundle_dir / "lazy" / "root"
    assert repr(loaded) == "LazyResult(unloaded)"
    assert _CountingCodec.load_calls == 0

    first = loaded.load()
    second = loaded.load()

    assert isinstance(first, _CountingValue)
    assert first.value == 9
    assert second is first
    assert loaded.is_loaded
    assert repr(loaded) == "LazyResult(_CountingValue)"
    assert _CountingCodec.load_calls == 1


def test_lazy_result_uses_declared_inner_annotated_codec(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = LazyResult(np.arange(4, dtype=np.int64))

    _save_result_bundle(
        value,
        bundle_dir,
        declared_type=LazyResult[Annotated[Any, NumpyNpyCodec]],
        registry=ResultRegistry.default(),
    )

    assert (bundle_dir / "lazy" / "root" / "artifacts" / "root" / "data.npy").exists()
    manifest = json.loads((bundle_dir / "lazy" / "root" / "manifest.json").read_text())
    assert manifest["$furu"]["|kind"] == "external"
    assert manifest["$furu"]["codec"] == NumpyNpyCodec._codec_id()


def test_nested_lazy_result_exposes_nested_persisted_path(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    registry = ResultRegistry.default().with_codec(_CountingCodec)
    value = {"outer": {"inner": LazyResult(_CountingValue(12))}}

    _save_result_bundle(value, bundle_dir, registry=registry)
    loaded = load_result_bundle(bundle_dir)

    assert isinstance(loaded, dict)
    loaded_dict = cast(dict[str, Any], loaded)
    outer = cast(dict[str, Any], loaded_dict["outer"])
    lazy = cast(LazyResult[_CountingValue], outer["inner"])
    assert lazy.path == bundle_dir / "lazy" / "outer" / "inner"
    assert lazy.path.joinpath("artifacts", "root", "value.txt").exists()
    assert lazy.load().value == 12


def test_nested_lazy_result_round_trips_inside_supported_structures(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    registry = ResultRegistry.default().with_codec(_CountingCodec)
    _CountingCodec.load_calls = 0
    value = {
        "items": [
            LazyResult(_CountingValue(1)),
            {"inner": LazyResult((Path("x"), 2))},
        ]
    }

    _save_result_bundle(value, bundle_dir, registry=registry)
    loaded = load_result_bundle(bundle_dir)

    assert isinstance(loaded, dict)
    loaded_dict = cast(dict[str, Any], loaded)
    items = loaded_dict["items"]
    assert isinstance(items, list)
    first = cast(LazyResult[_CountingValue], items[0])
    second_container = cast(dict[str, Any], items[1])
    second = cast(LazyResult[tuple[Path, int]], second_container["inner"])
    assert isinstance(first, LazyResult)
    assert isinstance(second, LazyResult)
    assert not first.is_loaded
    assert not second.is_loaded
    assert _CountingCodec.load_calls == 0

    assert first.load().value == 1
    assert second.load() == (Path("x"), 2)
    assert _CountingCodec.load_calls == 1


def test_load_result_bundle_rejects_lazy_path_escape(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "lazy").mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"$furu": {"|kind": "lazy", "path": "../outside"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes"):
        load_result_bundle(bundle_dir)


def test_load_result_bundle_rejects_lazy_without_nested_manifest(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    nested_dir = bundle_dir / "lazy" / "root"
    nested_dir.mkdir(parents=True)
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"$furu": {"|kind": "lazy", "path": "lazy/root"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nested manifest missing"):
        load_result_bundle(bundle_dir)
