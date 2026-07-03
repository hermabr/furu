from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, ClassVar, cast

import pytest
from pydantic import BaseModel, ConfigDict

import furu
from furu import Ref, Spec
from furu.storage._layout import data_dir_in, result_dir_in, result_manifest_path_in
from furu._declared_types import child_declared_type
from furu.result.bundle import (
    _DumpState,
    _save_result_bundle as _save_result_bundle_impl,
    load_result_bundle as load_result_bundle_impl,
)
from furu.result.codec import (
    Codec,
    CodecMeta,
    NumpyNpyCodec,
    PolarsParquetCodec,
)

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")


def _save_result_bundle(
    value: object,
    bundle_dir: Path,
    *,
    declared_type: object = Any,
    result_codecs: tuple[type[Codec], ...],
) -> _DumpState:
    return _save_result_bundle_impl(
        value,
        bundle_dir,
        declared_type=declared_type,
        result_codecs=result_codecs,
        data_dir=data_dir_in(bundle_dir.parent),
    )


def load_result_bundle(bundle_dir: Path, *, declared_type: object = Any) -> object:
    return load_result_bundle_impl(
        bundle_dir,
        data_dir=data_dir_in(bundle_dir.parent),
        declared_type=declared_type,
    )


_CHILD_DECLARED_TYPE_NAMESPACE: dict[str, object] = {
    "Annotated": Annotated,
    "Any": Any,
    "Codec": Codec,
    "Ellipsis": Ellipsis,
    "NumpyNpyCodec": NumpyNpyCodec,
    "Path": Path,
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


class JsonResult(Spec[dict[str, object]]):
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

    assert obj.status == "missing"
    obj.create()
    assert obj.status == "done"


def test_load_existing_returns_persisted_result() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    obj.create()

    assert obj.load_existing() == {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }


class ScalarResult(Spec[int]):
    def create(self) -> int:
        return 5


def test_scalar_root_manifest_is_just_the_value() -> None:
    obj = ScalarResult()

    assert obj.create() == 5
    text = result_manifest_path_in(obj._base_dir).read_text()
    assert json.loads(text) == 5


class PathResult(Spec[dict[str, Path]]):
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

    _save_result_bundle(value, bundle_dir, result_codecs=())

    assert load_result_bundle(bundle_dir) == value


def test_tuple_set_and_frozenset_round_trip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = {
        "tuple": (1, "x", Path("model.bin")),
        "set": {3, 1, 2},
        "frozenset": frozenset({"b", "a"}),
    }

    _save_result_bundle(value, bundle_dir, result_codecs=())

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


def test_set_of_values_without_value_based_repr_is_rejected(tmp_path: Path) -> None:
    class AddressRepr:
        __hash__ = object.__hash__

    bundle_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="no value-based repr"):
        _save_result_bundle(
            {AddressRepr(), AddressRepr()}, bundle_dir, result_codecs=()
        )


def test_tuple_root_value_uses_furu_wrapper(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = (1, 2, 3)

    _save_result_bundle(value, bundle_dir, result_codecs=())

    assert load_result_bundle(bundle_dir) == value
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest == {"$furu": {"|kind": "tuple", "items": [1, 2, 3]}}


class NonFiniteFloatResult(Spec[dict[str, float]]):
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


class _CountingCodec(Codec[_CountingValue]):
    auto_register: ClassVar[bool] = False
    dump_calls: ClassVar[int] = 0
    load_calls: ClassVar[int] = 0

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _CountingValue)

    def save(
        self, value: _CountingValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        type(self).dump_calls += 1
        artifact_directory.joinpath("value.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )
        return {}

    def load(
        self, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _CountingValue:
        type(self).load_calls += 1
        return _CountingValue(
            int(artifact_directory.joinpath("value.txt").read_text(encoding="utf-8"))
        )


class _OtherCountingCodec(Codec[_CountingValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _CountingValue)

    def save(
        self, value: _CountingValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("other.txt").write_text("x", encoding="utf-8")
        return {}

    def load(
        self, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _CountingValue:
        artifact_directory.joinpath("other.txt").read_text(encoding="utf-8")
        return _CountingValue(0)


class _CustomNumpyCodec(Codec[Any]):
    auto_register: ClassVar[bool] = False
    file_name: ClassVar[str] = "custom.npy"

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, np.ndarray)

    def save(self, value: Any, artifact_directory: Path) -> Mapping[str, object]:
        np.save(artifact_directory / self.file_name, value, allow_pickle=False)
        return {}

    def load(self, metadata: Mapping[str, object], artifact_directory: Path) -> Any:
        return np.load(artifact_directory / self.file_name, allow_pickle=False)


class _RegistryNumpyCodec(_CustomNumpyCodec):
    auto_register: ClassVar[bool] = False
    file_name: ClassVar[str] = "registry.npy"


class _AutoRegisteredValue:
    def __init__(self, value: int) -> None:
        self.value = value


class _AutoRegisteredValueCodec(Codec[_AutoRegisteredValue]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredValue)

    def save(
        self, value: _AutoRegisteredValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("auto.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )
        return {}

    def load(
        self, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(
            int(artifact_directory.joinpath("auto.txt").read_text(encoding="utf-8"))
        )


class _CoreRegistryAutoValueCodec(Codec[_AutoRegisteredValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredValue)

    def save(
        self, value: _AutoRegisteredValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("registry.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )
        return {}

    def load(
        self, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(
            int(artifact_directory.joinpath("registry.txt").read_text(encoding="utf-8"))
        )


class _AutoRegisteredArray(np.ndarray):
    pass


class _AutoRegisteredArrayCodec(Codec[Any]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredArray)

    def save(self, value: Any, artifact_directory: Path) -> Mapping[str, object]:
        np.save(
            artifact_directory / "auto.npy",
            value.view(np.ndarray),
            allow_pickle=False,
        )
        return {}

    def load(self, metadata: Mapping[str, object], artifact_directory: Path) -> Any:
        return np.load(artifact_directory / "auto.npy", allow_pickle=False)


class _OptOutRegisteredValue:
    pass


class _OptOutRegisteredValueCodec(Codec[_OptOutRegisteredValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _OptOutRegisteredValue)

    def save(
        self, value: _OptOutRegisteredValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("manual.txt").write_text("", encoding="utf-8")
        return {}

    def load(
        self, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _OptOutRegisteredValue:
        artifact_directory.joinpath("manual.txt").read_text(encoding="utf-8")
        return _OptOutRegisteredValue()


class _DataDirPathValue:
    def __init__(self, path: Path) -> None:
        self.path = path


class _DataDirPathCodec(Codec[_DataDirPathValue]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _DataDirPathValue)

    def save(
        self, value: _DataDirPathValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        return {"path": value.path}

    def load(
        self, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _DataDirPathValue:
        return _DataDirPathValue(cast(Path, metadata["path"]))


class RegistryAutoRegisteredValueResult(Spec[_AutoRegisteredValue]):
    @property
    def result_codecs(self) -> tuple[type[Codec], ...]:
        return (_CoreRegistryAutoValueCodec,)

    def create(self) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(10)


class AnnotatedAutoRegisteredValueResult(
    Spec[Annotated[_AutoRegisteredValue, _CoreRegistryAutoValueCodec]]
):
    def create(self) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(11)


def test_codec_id_is_derived_from_class_identity() -> None:
    assert _CountingCodec._codec_id() == (
        f"{_CountingCodec.__module__}.{_CountingCodec.__qualname__}"
    )


def test_result_codec_meta_find_codec_uses_result_codecs() -> None:
    first = (_CountingCodec,)
    second = (_CountingCodec, _OtherCountingCodec)

    assert CodecMeta.find_codec(_CountingValue(1), ()) is None
    assert CodecMeta.find_codec(_CountingValue(1), first) is _CountingCodec
    with pytest.raises(TypeError, match="result codecs matched multiple codecs"):
        CodecMeta.find_codec(_CountingValue(1), second)


def test_user_defined_codec_is_auto_registered(tmp_path: Path) -> None:
    assert (
        CodecMeta.find_codec(_AutoRegisteredValue(1), ())
        is _AutoRegisteredValueCodec
    )

    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        _AutoRegisteredValue(3),
        bundle_dir,
        result_codecs=(),
    )

    artifact_dir = bundle_dir / "artifacts" / "root"
    assert (artifact_dir / "auto.txt").exists()
    assert not (artifact_dir / "registry.txt").exists()
    loaded = load_result_bundle(bundle_dir)
    assert isinstance(loaded, _AutoRegisteredValue)
    assert loaded.value == 3


def test_auto_register_false_opts_out_of_auto_registered_codecs(
    tmp_path: Path,
) -> None:
    assert CodecMeta.find_codec(_OptOutRegisteredValue(), ()) is None

    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        _OptOutRegisteredValue(),
        bundle_dir,
        result_codecs=(_OptOutRegisteredValueCodec,),
    )

    assert (bundle_dir / "artifacts" / "root" / "manual.txt").exists()
    loaded = load_result_bundle(bundle_dir)
    assert isinstance(loaded, _OptOutRegisteredValue)


def test_auto_registered_codec_takes_priority_over_builtin_codec(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    value = np.arange(3, dtype=np.int64).view(_AutoRegisteredArray)

    _save_result_bundle(value, bundle_dir, result_codecs=())

    artifact_dir = bundle_dir / "artifacts" / "root"
    assert (artifact_dir / "auto.npy").exists()
    assert not (artifact_dir / "data.npy").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["$furu"]["codec"] == _AutoRegisteredArrayCodec._codec_id()


def test_task_result_codecs_take_priority_over_auto_registered_codec() -> None:
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


def test_codec_defined_after_default_codec_layers_cache_is_auto_registered() -> None:
    class LateAutoRegisteredValue:
        pass

    assert CodecMeta.find_codec(LateAutoRegisteredValue(), ()) is None

    class LateAutoRegisteredCodec(Codec[LateAutoRegisteredValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, LateAutoRegisteredValue)

        def save(
            self, value: LateAutoRegisteredValue, artifact_directory: Path
        ) -> Mapping[str, object]:
            artifact_directory.joinpath("late.txt").write_text("", encoding="utf-8")
            return {}

        def load(
            self, metadata: Mapping[str, object], artifact_directory: Path
        ) -> LateAutoRegisteredValue:
            artifact_directory.joinpath("late.txt").read_text(encoding="utf-8")
            return LateAutoRegisteredValue()

    assert (
        CodecMeta.find_codec(LateAutoRegisteredValue(), ())
        is LateAutoRegisteredCodec
    )


def test_explicit_registry_sees_later_auto_registered_codec() -> None:
    class LateExplicitRegistryAutoValue:
        pass

    result_codecs = (_CountingCodec,)

    assert (
        CodecMeta.find_codec(LateExplicitRegistryAutoValue(), result_codecs)
        is None
    )

    class LateExplicitRegistryAutoCodec(Codec[LateExplicitRegistryAutoValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, LateExplicitRegistryAutoValue)

        def save(
            self, value: LateExplicitRegistryAutoValue, artifact_directory: Path
        ) -> Mapping[str, object]:
            artifact_directory.joinpath("late-explicit.txt").write_text(
                "",
                encoding="utf-8",
            )
            return {}

        def load(
            self, metadata: Mapping[str, object], artifact_directory: Path
        ) -> LateExplicitRegistryAutoValue:
            artifact_directory.joinpath("late-explicit.txt").read_text(
                encoding="utf-8"
            )
            return LateExplicitRegistryAutoValue()

    assert (
        CodecMeta.find_codec(LateExplicitRegistryAutoValue(), result_codecs)
        is LateExplicitRegistryAutoCodec
    )


def test_auto_registered_codecs_must_not_be_ambiguous() -> None:
    class AutoAmbiguousValue:
        pass

    class FirstAutoAmbiguousCodec(Codec[AutoAmbiguousValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, AutoAmbiguousValue)

        def save(
            self, value: AutoAmbiguousValue, artifact_directory: Path
        ) -> Mapping[str, object]:
            artifact_directory.joinpath("first.txt").write_text("", encoding="utf-8")
            return {}

        def load(
            self, metadata: Mapping[str, object], artifact_directory: Path
        ) -> AutoAmbiguousValue:
            artifact_directory.joinpath("first.txt").read_text(encoding="utf-8")
            return AutoAmbiguousValue()

    class SecondAutoAmbiguousCodec(Codec[AutoAmbiguousValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, AutoAmbiguousValue)

        def save(
            self, value: AutoAmbiguousValue, artifact_directory: Path
        ) -> Mapping[str, object]:
            artifact_directory.joinpath("second.txt").write_text("", encoding="utf-8")
            return {}

        def load(
            self, metadata: Mapping[str, object], artifact_directory: Path
        ) -> AutoAmbiguousValue:
            artifact_directory.joinpath("second.txt").read_text(encoding="utf-8")
            return AutoAmbiguousValue()

    with pytest.raises(TypeError) as exc_info:
        CodecMeta.find_codec(AutoAmbiguousValue(), ())

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


class AnnotatedArrayResult(Spec[AnnotatedArrayOutput]):
    def create(self) -> AnnotatedArrayOutput:
        return AnnotatedArrayOutput(weights=np.arange(3, dtype=np.int64))


class GenericAnnotatedArrayBase[T](Spec[T]):
    def create(self) -> T:
        return np.arange(3, dtype=np.int64)


class GenericAnnotatedArrayResult(
    GenericAnnotatedArrayBase[Annotated[Any, NumpyNpyCodec]]
):
    pass


class StrictAnnotatedArrayOutput(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    weights: Annotated[np.ndarray[Any, Any], NumpyNpyCodec]


class StrictAnnotatedArrayResult(Spec[StrictAnnotatedArrayOutput]):
    def create(self) -> StrictAnnotatedArrayOutput:
        return StrictAnnotatedArrayOutput(weights=np.arange(3, dtype=np.int64))


def test_annotated_codec_selects_artifact_storage() -> None:
    obj = AnnotatedArrayResult()
    loaded = obj.create()

    assert isinstance(loaded, AnnotatedArrayOutput)
    assert np.array_equal(loaded.weights, np.arange(3, dtype=np.int64))
    assert (
        result_dir_in(obj._base_dir) / "artifacts" / "weights" / "data.npy"
    ).exists()


def test_generic_furu_base_with_annotated_result_codec_is_rejected() -> None:
    obj = GenericAnnotatedArrayResult()

    with pytest.raises(TypeError, match="concrete result type directly as Spec"):
        obj.create()


def test_strict_pydantic_annotated_codec_selects_artifact_storage() -> None:
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
class RefArrayOutput:
    weights: Ref[Any]


class RefArrayResult(Spec[RefArrayOutput]):
    def create(self) -> RefArrayOutput:
        return RefArrayOutput(weights=furu.ref(np.arange(4)))


@dataclass(frozen=True)
class RefCountingOutput:
    value: Ref[_CountingValue]


class RefCountingResult(Spec[RefCountingOutput]):
    def create(self) -> RefCountingOutput:
        return RefCountingOutput(
            value=furu.ref(_CountingValue(9), codec=_CountingCodec)
        )


class DataDirPathResult(Spec[dict[str, _DataDirPathValue]]):
    @property
    def result_codecs(self) -> tuple[type[Codec], ...]:
        return (_DataDirPathCodec,)

    def create(self) -> dict[str, _DataDirPathValue]:
        path = self.directory.data / "data.zarr"
        path.mkdir()
        value = _DataDirPathValue(path)
        return {
            "first": value,
            "second": cast(
                _DataDirPathValue, furu.ref(value, codec=_DataDirPathCodec)
            ),
        }


def test_ref_field_resolves_codec_from_registry_and_loads_on_demand() -> None:
    obj = RefArrayResult()
    created = obj.create()

    assert isinstance(created, RefArrayOutput)
    assert isinstance(created.weights, Ref)
    assert np.array_equal(created.weights.load(), np.arange(4))
    assert (
        result_dir_in(obj._base_dir) / "artifacts" / "weights" / "data.npy"
    ).exists()

    loaded_again = obj.create()
    assert isinstance(loaded_again, RefArrayOutput)
    assert isinstance(loaded_again.weights, Ref)
    assert np.array_equal(loaded_again.weights.load(), np.arange(4))


def test_ref_rebinds_to_storage_after_publish() -> None:
    _CountingCodec.dump_calls = 0
    _CountingCodec.load_calls = 0
    obj = RefCountingResult()

    created = obj.create()

    assert isinstance(created.value, Ref)
    assert _CountingCodec.dump_calls == 1
    # Rebound, not reloaded: publishing must not eagerly read the artifact back.
    assert _CountingCodec.load_calls == 0

    first = created.value.load()
    second = created.value.load()

    assert first.value == 9
    assert second is first
    assert _CountingCodec.load_calls == 1


def test_ref_without_resolvable_codec_raises_at_call_site() -> None:
    with pytest.raises(TypeError, match=r"furu\.ref\(\) found no codec") as exc_info:
        furu.ref([1, 2, 3])

    message = str(exc_info.value)
    assert "codec=" in message
    assert "eager" in message


@dataclass(frozen=True)
class ConflictingRefOutput:
    weights: Annotated[Ref[Any], NumpyNpyCodec]


class ConflictingRefResult(Spec[ConflictingRefOutput]):
    def create(self) -> ConflictingRefOutput:
        return ConflictingRefOutput(
            weights=furu.ref(_CountingValue(1), codec=_CountingCodec)
        )


def test_ref_codec_conflicts_with_different_annotated_codec() -> None:
    with pytest.raises(TypeError, match="Conflicting codecs"):
        ConflictingRefResult().create()


def test_codec_metadata_path_round_trips_shared_data_dir_path() -> None:
    obj = DataDirPathResult()
    loaded = obj.create()
    data_path = obj.directory.data / "data.zarr"

    assert loaded["first"].path.resolve() == data_path.resolve()
    # The creating run keeps the handle it put in, rebound to storage.
    second = cast(Ref[_DataDirPathValue], loaded["second"])
    assert isinstance(second, Ref)
    assert second.load().path.resolve() == data_path.resolve()

    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest["first"]["$furu"]["metadata"] == {
        "path": {"$furu": {"|kind": "path", "value": "data.zarr"}}
    }
    assert not (
        result_dir_in(obj._base_dir) / "artifacts" / "first" / "data.zarr"
    ).exists()

    loaded_again = obj.load_existing()

    assert loaded_again["first"].path == data_path.resolve()
    # The field is declared eager, so the Ref spelling on save still loads
    # eagerly on read.
    assert isinstance(loaded_again["second"], _DataDirPathValue)
    assert loaded_again["second"].path == data_path.resolve()


def test_codec_metadata_path_outside_data_dir_raises_at_save(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    outside_path = tmp_path / "outside"

    with pytest.raises(ValueError, match="must live inside the data dir"):
        _save_result_bundle(
            _DataDirPathValue(outside_path),
            bundle_dir,
            result_codecs=(_DataDirPathCodec,),
        )


def test_codec_metadata_rejects_load_path_outside_data_dir(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "object" / "result"
    artifact_dir = result_dir / "artifacts" / "root"
    artifact_dir.mkdir(parents=True)

    def write_manifest(path_value: str) -> None:
        (result_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "$furu": {
                        "|kind": "artifact",
                        "codec": _DataDirPathCodec._codec_id(),
                        "path": "artifacts/root",
                        "metadata": {
                            "path": {"$furu": {"|kind": "path", "value": path_value}}
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

    write_manifest("/tmp/outside")
    with pytest.raises(ValueError, match="must be relative"):
        load_result_bundle(result_dir)

    write_manifest("../outside")
    with pytest.raises(ValueError, match="escapes data dir"):
        load_result_bundle(result_dir)


class RegistryCountingResult(Spec[_CountingValue]):
    @property
    def result_codecs(self) -> tuple[type[Codec], ...]:
        return (_CountingCodec,)

    def create(self) -> _CountingValue:
        return _CountingValue(8)


class AmbiguousRegistryCountingResult(Spec[_CountingValue]):
    @property
    def result_codecs(self) -> tuple[type[Codec], ...]:
        return (_CountingCodec, _OtherCountingCodec)

    def create(self) -> _CountingValue:
        return _CountingValue(8)


def test_task_result_codecs_are_used_for_save_inference_only() -> None:
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


def test_task_result_codecs_must_not_be_ambiguous() -> None:
    with pytest.raises(TypeError, match="result codecs matched multiple codecs"):
        AmbiguousRegistryCountingResult().create()


class UnsupportedRootResult(Spec[object]):
    def create(self) -> object:
        return _CustomTensor()


def test_unsupported_custom_object_fails_with_root_path(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle(_CustomTensor(), bundle_dir, result_codecs=())
    msg = str(exc_info.value)
    assert "<root>" in msg
    assert "_CustomTensor" in msg


def test_unsupported_nested_path_includes_padded_index(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    layers = [{} for _ in range(10)]
    layers[3] = {"weights": _CustomTensor()}

    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle({"layers": layers}, bundle_dir, result_codecs=())
    assert "layers/03/weights" in str(exc_info.value)


def test_reserved_furu_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="reserved"):
        _save_result_bundle({"$furu": "user data"}, bundle_dir, result_codecs=())


def test_non_string_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="must be strings"):
        _save_result_bundle({1: "x"}, bundle_dir, result_codecs=())


def test_unsafe_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle({"bad/key": "x"}, bundle_dir, result_codecs=())
    assert "artifact path segment" in str(exc_info.value)


def test_empty_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle({"": "x"}, bundle_dir, result_codecs=())
    assert "artifact path segment" in str(exc_info.value)


def test_dotdot_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="artifact path segment"):
        _save_result_bundle({"..": "x"}, bundle_dir, result_codecs=())


@dataclass(frozen=True)
class TrainOutput:
    metrics: dict[str, float]
    values: list[int]


class DataclassResult(Spec[TrainOutput]):
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
    _save_result_bundle(DataclassWithPostInit(value=3), bundle_dir, result_codecs=())

    loaded = load_result_bundle(bundle_dir)

    assert loaded == DataclassWithPostInit(value=3)


def test_dataclass_load_reports_constructor_error_with_path(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        {"result": DataclassWithPostInit(value=3)},
        bundle_dir,
        result_codecs=(),
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
        result_codecs=(),
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


class NestedDataclassResult(Spec[NestedOuter]):
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


class PydanticResult(Spec[TrainOutputModel]):
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
        result_codecs=(),
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
        result_codecs=(),
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


class NestedPydanticResult(Spec[NestedPydanticOuter]):
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


class NumpyResult(Spec[dict[str, object]]):
    def create(self) -> dict[str, object]:
        return {"weights": np.arange(10, dtype=np.float32)}


class RegistryNumpyResult(Spec[Any]):
    @property
    def result_codecs(self) -> tuple[type[Codec], ...]:
        return (_RegistryNumpyCodec,)

    def create(self) -> Any:
        return np.arange(10, dtype=np.float32)


class LargeNumpyResult(Spec[np.ndarray]):
    def create(self) -> np.ndarray[Any, Any]:
        length = NumpyNpyCodec.memmap_threshold_bytes // 8 + 1
        return np.arange(length, dtype=np.float64)


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
    assert manifest["weights"]["$furu"]["|kind"] == "artifact"
    assert manifest["weights"]["$furu"]["codec"] == (
        f"{NumpyNpyCodec.__module__}.{NumpyNpyCodec.__qualname__}"
    )
    assert manifest["weights"]["$furu"]["path"] == "artifacts/weights"
    assert manifest["weights"]["$furu"]["metadata"] == {}


def test_result_codecs_take_priority_over_builtin_codec() -> None:
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


def test_array_over_memmap_threshold_is_memmap_backed_from_the_creating_run() -> None:
    obj = LargeNumpyResult()
    expected_file = result_dir_in(obj._base_dir) / "artifacts" / "root" / "data.npy"
    expected = np.arange(
        NumpyNpyCodec.memmap_threshold_bytes // 8 + 1, dtype=np.float64
    )

    first = obj.create()
    second = obj.load_existing()

    assert isinstance(first, np.ndarray)
    assert isinstance(first, np.memmap)
    assert isinstance(second, np.memmap)
    assert Path(first.filename).resolve() == expected_file.resolve()
    assert Path(second.filename).resolve() == expected_file.resolve()
    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)


def test_array_under_memmap_threshold_reloads_as_plain_array() -> None:
    obj = NumpyResult()

    created = obj.create()

    weights = cast(Any, created["weights"])
    assert isinstance(weights, np.ndarray)
    assert not isinstance(weights, np.memmap)


class PolarsResult(Spec[dict[str, object]]):
    def create(self) -> dict[str, object]:
        return {"frame": pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})}


class PolarsRefResult(Spec[Ref[Any]]):
    def create(self) -> Ref[Any]:
        return furu.ref(pl.DataFrame({"x": [1, 2]}))


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
    assert manifest["frame"]["$furu"]["|kind"] == "artifact"
    assert manifest["frame"]["$furu"]["codec"] == (
        f"{PolarsParquetCodec.__module__}.{PolarsParquetCodec.__qualname__}"
    )
    assert manifest["frame"]["$furu"]["path"] == "artifacts/frame"


def test_polars_whole_value_routes_through_registry_and_loads_eagerly(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    frame = pl.DataFrame({"x": [1, 2, 3]})

    _save_result_bundle(frame, bundle_dir, result_codecs=())
    loaded = load_result_bundle(bundle_dir)

    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(frame)


def test_polars_ref_result_round_trips() -> None:
    obj = PolarsRefResult()

    created = obj.create()
    loaded = obj.load_existing()

    assert isinstance(created, Ref)
    assert created.load().equals(pl.DataFrame({"x": [1, 2]}))
    assert isinstance(loaded, Ref)
    assert loaded.load().equals(pl.DataFrame({"x": [1, 2]}))


def test_numpy_object_dtype_is_rejected(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="allow_pickle=False"):
        _save_result_bundle(
            {"weights": np.array([object()], dtype=object)},
            bundle_dir,
            result_codecs=(),
        )


class NestedNumpyResult(Spec[dict[str, list[dict[str, object]]]]):
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

    _save_result_bundle({"layers": layers}, bundle_dir, result_codecs=())

    expected = bundle_dir / "artifacts" / "layers" / "003" / "weights" / "data.npy"
    assert expected.exists()


def test_numpy_root_value_uses_root_artifact_dir(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(np.arange(5, dtype=np.int64), bundle_dir, result_codecs=())

    assert (bundle_dir / "artifacts" / "root" / "data.npy").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["$furu"]["|kind"] == "artifact"
    assert manifest["$furu"]["path"] == "artifacts/root"

    loaded = load_result_bundle(bundle_dir)
    assert np.array_equal(loaded, np.arange(5, dtype=np.int64))


@dataclass(frozen=True)
class MixedOutput:
    metrics: dict[str, float]
    values: list[int]


class MixedResult(Spec[dict[str, object]]):
    def create(self) -> dict[str, object]:
        return {
            "result": MixedOutput(metrics={"loss": 0.5}, values=[1, 2]),
            "weights": np.arange(4, dtype=np.float32),
            "labels": ["cat", "dog"],
        }


def test_mixed_dataclass_artifact_and_json_round_trip() -> None:
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
        _save_result_bundle({"x": 1}, bundle_dir, result_codecs=())


def test_private_save_result_bundle_writes_manifest_last(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        {"weights": np.arange(2, dtype=np.float32)},
        bundle_dir,
        result_codecs=(),
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
                    "|kind": "artifact",
                    "codec": f"{NumpyNpyCodec.__module__}.{NumpyNpyCodec.__qualname__}",
                    "path": "../../../etc/passwd",
                    "metadata": {},
                }
            }
        )
    )

    with pytest.raises(ValueError, match="escapes"):
        load_result_bundle(bundle_dir)


def test_ref_created_directly_holds_its_value() -> None:
    value = _CountingValue(7)
    handle = furu.ref(value, codec=_CountingCodec)

    assert repr(handle) == "Ref(_CountingValue)"
    assert handle.load() is value


def test_root_ref_defers_cache_read_and_memoizes(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _CountingCodec.dump_calls = 0
    _CountingCodec.load_calls = 0

    _save_result_bundle(
        furu.ref(_CountingValue(9), codec=_CountingCodec),
        bundle_dir,
        result_codecs=(),
    )

    assert _CountingCodec.dump_calls == 1
    assert _CountingCodec.load_calls == 0
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest == {
        "$furu": {
            "|kind": "artifact",
            "codec": _CountingCodec._codec_id(),
            "path": "artifacts/root",
            "metadata": {},
        }
    }

    loaded = load_result_bundle(
        bundle_dir, declared_type=Ref[_CountingValue]
    )

    assert isinstance(loaded, Ref)
    assert repr(loaded) == "Ref(unloaded)"
    assert _CountingCodec.load_calls == 0

    first = loaded.load()
    second = loaded.load()

    assert isinstance(first, _CountingValue)
    assert first.value == 9
    assert second is first
    assert repr(loaded) == "Ref(_CountingValue)"
    assert _CountingCodec.load_calls == 1


def test_refs_round_trip_inside_supported_structures(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _CountingCodec.load_calls = 0
    value = {
        "items": [
            {"inner": furu.ref(_CountingValue(1), codec=_CountingCodec)},
            {"inner": furu.ref(_CountingValue(2), codec=_CountingCodec)},
        ]
    }

    _save_result_bundle(value, bundle_dir, result_codecs=())
    loaded = load_result_bundle(
        bundle_dir,
        declared_type=dict[str, list[dict[str, Ref[_CountingValue]]]],
    )

    assert isinstance(loaded, dict)
    loaded_dict = cast(dict[str, Any], loaded)
    items = loaded_dict["items"]
    assert isinstance(items, list)
    first = cast(Ref[_CountingValue], cast(dict[str, Any], items[0])["inner"])
    second = cast(Ref[_CountingValue], cast(dict[str, Any], items[1])["inner"])
    assert isinstance(first, Ref)
    assert isinstance(second, Ref)
    assert _CountingCodec.load_calls == 0

    assert first.load().value == 1
    assert second.load().value == 2
    assert _CountingCodec.load_calls == 2


class RootRefResult(Spec[Ref[_CountingValue]]):
    def create(self) -> Ref[_CountingValue]:
        return furu.ref(_CountingValue(21), codec=_CountingCodec)


def test_ref_as_the_whole_result_rebinds_and_rehydrates() -> None:
    _CountingCodec.load_calls = 0
    obj = RootRefResult()

    created = obj.create()

    assert isinstance(created, Ref)
    assert _CountingCodec.load_calls == 0
    assert created.load().value == 21
    assert _CountingCodec.load_calls == 1

    loaded = obj.load_existing()

    assert isinstance(loaded, Ref)
    assert loaded.load().value == 21


@dataclass(frozen=True)
class MixedRefOutput:
    loss: float
    weights: np.ndarray
    history: Ref[_CountingValue]


class MixedRefResult(Spec[MixedRefOutput]):
    def create(self) -> MixedRefOutput:
        return MixedRefOutput(
            loss=0.25,
            weights=np.arange(4, dtype=np.float32),
            history=furu.ref(_CountingValue(3), codec=_CountingCodec),
        )


def test_mixed_eager_ref_and_inline_fields_round_trip() -> None:
    _CountingCodec.load_calls = 0
    obj = MixedRefResult()

    obj.create()
    loaded = obj.load_existing()

    assert isinstance(loaded, MixedRefOutput)
    assert loaded.loss == 0.25
    assert isinstance(loaded.weights, np.ndarray)
    assert np.array_equal(loaded.weights, np.arange(4, dtype=np.float32))
    assert isinstance(loaded.history, Ref)
    assert _CountingCodec.load_calls == 0
    assert loaded.history.load().value == 3

    # The inline field is readable straight from the manifest.
    manifest = json.loads(result_manifest_path_in(obj._base_dir).read_text())
    assert manifest["$furu"]["|fields"]["loss"] == 0.25


def test_eager_and_ref_spellings_are_byte_identical_on_disk(tmp_path: Path) -> None:
    eager_dir = tmp_path / "eager" / "bundle"
    ref_dir = tmp_path / "ref" / "bundle"
    array = np.arange(4, dtype=np.int64)

    _save_result_bundle({"weights": array}, eager_dir, result_codecs=())
    _save_result_bundle({"weights": furu.ref(array)}, ref_dir, result_codecs=())

    assert (eager_dir / "manifest.json").read_bytes() == (
        ref_dir / "manifest.json"
    ).read_bytes()
    assert (eager_dir / "artifacts" / "weights" / "data.npy").read_bytes() == (
        ref_dir / "artifacts" / "weights" / "data.npy"
    ).read_bytes()

    # The declared type alone decides rehydration, in both directions.
    eager_from_ref_bundle = cast(
        dict[str, Any],
        load_result_bundle(ref_dir, declared_type=dict[str, np.ndarray]),
    )
    ref_from_eager_bundle = cast(
        dict[str, Any],
        load_result_bundle(eager_dir, declared_type=dict[str, Ref[np.ndarray]]),
    )

    assert isinstance(eager_from_ref_bundle["weights"], np.ndarray)
    assert isinstance(ref_from_eager_bundle["weights"], Ref)
    assert np.array_equal(eager_from_ref_bundle["weights"], array)
    assert np.array_equal(ref_from_eager_bundle["weights"].load(), array)
