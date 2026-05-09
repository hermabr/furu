from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pydantic import BaseModel, ConfigDict

from furu import Furu
from furu.result import (
    LazyResult,
    load_result_bundle,
    save_result_bundle,
)
from furu.result.codec import (
    _DEFAULT_CODECS,
    NumpyNpyCodec,
    PolarsParquetCodec,
    ResultCodec,
)

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")


class JsonResult(Furu[dict[str, object]]):
    create_calls: ClassVar[list[int]] = []

    def _create(self) -> dict[str, object]:
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

    result = obj.load_or_create()
    assert result == expected

    assert (obj._result_dir / "manifest.json").exists()
    assert not (obj._result_dir / "artifacts").exists()

    manifest = json.loads((obj._result_dir / "manifest.json").read_text())
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

    assert obj.load_or_create() == expected
    assert obj.load_or_create() == expected
    assert obj._result_manifest_path == (obj._result_dir / "manifest.json")
    assert len(JsonResult.create_calls) == 1


def test_status_is_completed_after_first_run() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    assert obj.status() == "missing"
    obj.load_or_create()
    assert obj.status() == "completed"


def test_try_load_returns_persisted_result() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    obj.load_or_create()

    assert obj.try_load() == {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }


class ScalarResult(Furu[int]):
    def _create(self) -> int:
        return 5


def test_scalar_root_manifest_is_just_the_value() -> None:
    obj = ScalarResult()

    assert obj.load_or_create() == 5
    text = (obj._result_dir / "manifest.json").read_text()
    assert json.loads(text) == 5


class PathResult(Furu[dict[str, Path]]):
    def _create(self) -> dict[str, Path]:
        return {
            "relative": Path("outputs/model.bin"),
            "absolute": Path("/tmp/furu/model.bin"),
        }


def test_path_values_round_trip() -> None:
    obj = PathResult()

    result = obj.load_or_create()

    assert result == {
        "relative": Path("outputs/model.bin"),
        "absolute": Path("/tmp/furu/model.bin"),
    }

    manifest = json.loads((obj._result_dir / "manifest.json").read_text())
    assert manifest == {
        "relative": {
            "$furu": {
                "kind": "path",
                "value": "outputs/model.bin",
            }
        },
        "absolute": {
            "$furu": {
                "kind": "path",
                "value": "/tmp/furu/model.bin",
            }
        },
    }


def test_path_root_value_round_trips(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = Path("outputs/model.bin")

    save_result_bundle(value, bundle_dir)

    assert load_result_bundle(bundle_dir) == value


def test_tuple_set_and_frozenset_round_trip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = {
        "tuple": (1, "x", Path("model.bin")),
        "set": {3, 1, 2},
        "frozenset": frozenset({"b", "a"}),
    }

    save_result_bundle(value, bundle_dir)

    assert load_result_bundle(bundle_dir) == value
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["tuple"] == {
        "$furu": {
            "kind": "tuple",
            "items": [
                1,
                "x",
                {"$furu": {"kind": "path", "value": "model.bin"}},
            ],
        }
    }
    assert manifest["set"]["$furu"] == {"kind": "set", "items": [1, 2, 3]}
    assert manifest["frozenset"]["$furu"] == {
        "kind": "frozenset",
        "items": ["a", "b"],
    }


def test_tuple_root_value_uses_furu_wrapper(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = (1, 2, 3)

    save_result_bundle(value, bundle_dir)

    assert load_result_bundle(bundle_dir) == value
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest == {"$furu": {"kind": "tuple", "items": [1, 2, 3]}}


class NonFiniteFloatResult(Furu[dict[str, float]]):
    def _create(self) -> dict[str, float]:
        return {
            "nan": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
        }


def test_non_finite_floats_round_trip() -> None:
    obj = NonFiniteFloatResult()
    result = obj.load_or_create()

    assert math.isnan(result["nan"])
    assert result["pos_inf"] == float("inf")
    assert result["neg_inf"] == float("-inf")

    text = (obj._result_dir / "manifest.json").read_text()
    # Python's standard library writes these as JSON extensions.
    assert "NaN" in text
    assert "Infinity" in text
    assert "-Infinity" in text


class _CustomTensor:
    pass


class _CountingValue:
    def __init__(self, value: int) -> None:
        self.value = value


class _CountingCodec(ResultCodec):
    dump_calls: ClassVar[int] = 0
    load_calls: ClassVar[int] = 0

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _CountingValue)

    @classmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
    ) -> None:
        cls.dump_calls += 1
        artifact_dir.joinpath("value.txt").write_text(
            str(cast(_CountingValue, value).value),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, *, artifact_dir: Path) -> object:
        cls.load_calls += 1
        return _CountingValue(
            int(artifact_dir.joinpath("value.txt").read_text(encoding="utf-8"))
        )


def test_codec_id_is_derived_from_class_identity() -> None:
    assert _CountingCodec.codec_id() == (
        f"{_CountingCodec.__module__}.{_CountingCodec.__qualname__}"
    )


def test_codec_id_override_is_rejected() -> None:
    with pytest.raises(TypeError, match="must not override codec_id"):

        class InvalidCodec(ResultCodec):
            @classmethod
            def codec_id(cls) -> str:
                return "custom"

            @classmethod
            def matches(cls, value: object) -> bool:
                return False

            @classmethod
            def dump(
                cls,
                value: object,
                *,
                artifact_dir: Path,
            ) -> None:
                raise AssertionError("unreachable")

            @classmethod
            def load(cls, *, artifact_dir: Path) -> object:
                raise AssertionError("unreachable")


class UnsupportedRootResult(Furu[object]):
    def _create(self) -> object:
        return _CustomTensor()


def test_unsupported_custom_object_fails_with_root_path(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        save_result_bundle(_CustomTensor(), bundle_dir)
    msg = str(exc_info.value)
    assert "<root>" in msg
    assert "_CustomTensor" in msg


def test_unsupported_nested_path_includes_padded_index(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    layers = [{} for _ in range(10)]
    layers[3] = {"weights": _CustomTensor()}

    with pytest.raises(ValueError) as exc_info:
        save_result_bundle({"layers": layers}, bundle_dir)
    assert "layers/03/weights" in str(exc_info.value)


def test_reserved_furu_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="reserved"):
        save_result_bundle({"$furu": "user data"}, bundle_dir)


def test_non_string_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="must be strings"):
        save_result_bundle({1: "x"}, bundle_dir)


def test_unsafe_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        save_result_bundle({"bad/key": "x"}, bundle_dir)
    assert "artifact path segment" in str(exc_info.value)


def test_empty_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError) as exc_info:
        save_result_bundle({"": "x"}, bundle_dir)
    assert "artifact path segment" in str(exc_info.value)


def test_dotdot_dict_key_fails(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="artifact path segment"):
        save_result_bundle({"..": "x"}, bundle_dir)


@dataclass(frozen=True)
class TrainOutput:
    metrics: dict[str, float]
    values: list[int]


class DataclassResult(Furu[TrainOutput]):
    def _create(self) -> TrainOutput:
        return TrainOutput(metrics={"loss": 0.12}, values=[1, 2, 3])


def test_dataclass_round_trip() -> None:
    obj = DataclassResult()
    loaded = obj.load_or_create()
    assert isinstance(loaded, TrainOutput)
    assert loaded == TrainOutput(metrics={"loss": 0.12}, values=[1, 2, 3])

    manifest = json.loads((obj._result_dir / "manifest.json").read_text())
    assert manifest["$furu"]["kind"] == "dataclass"
    assert manifest["$furu"]["type"] == "test_result.TrainOutput"
    assert manifest["$furu"]["fields"] == {
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
    save_result_bundle(DataclassWithPostInit(value=3), bundle_dir)

    loaded = load_result_bundle(bundle_dir)

    assert loaded == DataclassWithPostInit(value=3)


def test_dataclass_load_reports_constructor_error_with_path(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    save_result_bundle({"result": DataclassWithPostInit(value=3)}, bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["result"]["$furu"]["fields"]["value"] = -1
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
    save_result_bundle(
        {"result": TrainOutput(metrics={"loss": 0.12}, values=[1])}, bundle_dir
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fields = manifest["result"]["$furu"]["fields"]
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
    def _create(self) -> NestedOuter:
        return NestedOuter(inner=NestedInner(value=42), label="root")


def test_nested_dataclass_round_trip() -> None:
    obj = NestedDataclassResult()
    loaded = obj.load_or_create()
    assert isinstance(loaded, NestedOuter)
    assert isinstance(loaded.inner, NestedInner)
    assert loaded == NestedOuter(inner=NestedInner(value=42), label="root")


class TrainOutputModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    metrics: dict[str, float]
    values: list[int]


class PydanticResult(Furu[TrainOutputModel]):
    def _create(self) -> TrainOutputModel:
        return TrainOutputModel(metrics={"loss": 0.12}, values=[1, 2, 3])


def test_pydantic_round_trip() -> None:
    obj = PydanticResult()
    loaded = obj.load_or_create()
    assert isinstance(loaded, TrainOutputModel)
    assert loaded.metrics == {"loss": 0.12}
    assert loaded.values == [1, 2, 3]


class ValidatedTrainOutputModel(BaseModel):
    value: int


def test_pydantic_load_uses_model_validate(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    save_result_bundle(ValidatedTrainOutputModel(value=1), bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["$furu"]["fields"]["value"] = "not an int"
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
    save_result_bundle(
        {"models": [TrainOutputModel(metrics={}, values=[])]}, bundle_dir
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fields = manifest["models"][0]["$furu"]["fields"]
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
    def _create(self) -> NestedPydanticOuter:
        return NestedPydanticOuter(
            metrics={"loss": 0.12, "accuracy": 0.94},
            items=[{"v": 1}, {"v": 2}, {"v": 3}],
        )


def test_pydantic_with_nested_structures_round_trips() -> None:
    obj = NestedPydanticResult()
    loaded = obj.load_or_create()
    assert isinstance(loaded, NestedPydanticOuter)
    assert loaded.metrics == {"loss": 0.12, "accuracy": 0.94}
    assert loaded.items == [{"v": 1}, {"v": 2}, {"v": 3}]


class NumpyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"weights": np.arange(10, dtype=np.float32)}


def test_numpy_array_round_trips() -> None:
    obj = NumpyResult()
    loaded = obj.load_or_create()

    assert (obj._result_dir / "artifacts" / "weights" / "data.npy").exists()
    assert isinstance(loaded, dict)
    weights = cast(Any, loaded["weights"])
    assert weights.dtype == np.float32
    assert np.array_equal(weights, np.arange(10, dtype=np.float32))

    manifest = json.loads((obj._result_dir / "manifest.json").read_text())
    assert manifest["weights"]["$furu"]["kind"] == "external"
    assert manifest["weights"]["$furu"]["codec"] == (
        f"{NumpyNpyCodec.__module__}.{NumpyNpyCodec.__qualname__}"
    )
    assert manifest["weights"]["$furu"]["path"] == "artifacts/weights"


class PolarsResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"frame": pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})}


def test_polars_dataframe_round_trips() -> None:
    obj = PolarsResult()
    loaded = obj.load_or_create()

    assert (obj._result_dir / "artifacts" / "frame" / "data.parquet").exists()
    assert isinstance(loaded, dict)
    frame = loaded["frame"]
    assert isinstance(frame, pl.DataFrame)
    assert frame.equals(pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}))

    manifest = json.loads((obj._result_dir / "manifest.json").read_text())
    assert manifest["frame"]["$furu"]["kind"] == "external"
    assert manifest["frame"]["$furu"]["codec"] == (
        f"{PolarsParquetCodec.__module__}.{PolarsParquetCodec.__qualname__}"
    )
    assert manifest["frame"]["$furu"]["path"] == "artifacts/frame"


def test_numpy_object_dtype_is_rejected(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="allow_pickle=False"):
        save_result_bundle(
            {"weights": np.array([object()], dtype=object)},
            bundle_dir,
        )


class NestedNumpyResult(Furu[dict[str, list[dict[str, object]]]]):
    def _create(self) -> dict[str, list[dict[str, object]]]:
        return {
            "layers": [{"weights": np.arange(i, dtype=np.float32)} for i in range(10)]
        }


def test_nested_numpy_paths_use_list_length_padded_indexes() -> None:
    obj = NestedNumpyResult()
    loaded = obj.load_or_create()

    layers_dir = obj._result_dir / "artifacts" / "layers"
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

    save_result_bundle({"layers": layers}, bundle_dir)

    expected = bundle_dir / "artifacts" / "layers" / "003" / "weights" / "data.npy"
    assert expected.exists()


def test_numpy_root_value_uses_root_artifact_dir(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    save_result_bundle(np.arange(5, dtype=np.int64), bundle_dir)

    assert (bundle_dir / "artifacts" / "root" / "data.npy").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["$furu"]["kind"] == "external"
    assert manifest["$furu"]["path"] == "artifacts/root"

    loaded = load_result_bundle(bundle_dir)
    assert np.array_equal(loaded, np.arange(5, dtype=np.int64))


@dataclass(frozen=True)
class MixedOutput:
    metrics: dict[str, float]
    values: list[int]


class MixedResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "result": MixedOutput(metrics={"loss": 0.5}, values=[1, 2]),
            "weights": np.arange(4, dtype=np.float32),
            "labels": ["cat", "dog"],
        }


def test_mixed_dataclass_external_and_json_round_trip() -> None:
    obj = MixedResult()
    loaded = obj.load_or_create()

    assert isinstance(loaded, dict)
    inner = loaded["result"]
    assert isinstance(inner, MixedOutput)
    assert inner == MixedOutput(metrics={"loss": 0.5}, values=[1, 2])
    assert np.array_equal(loaded["weights"], np.arange(4, dtype=np.float32))
    assert loaded["labels"] == ["cat", "dog"]


def test_save_result_bundle_refuses_existing_directory(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    with pytest.raises(FileExistsError):
        save_result_bundle({"x": 1}, bundle_dir)


def test_save_result_bundle_writes_manifest_last(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    save_result_bundle({"weights": np.arange(2, dtype=np.float32)}, bundle_dir)

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
                    "kind": "external",
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


def test_root_lazy_result_defers_cache_read_and_memoizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _CountingCodec.dump_calls = 0
    _CountingCodec.load_calls = 0
    monkeypatch.setitem(_DEFAULT_CODECS, _CountingCodec.codec_id(), _CountingCodec)

    save_result_bundle(LazyResult(_CountingValue(9)), bundle_dir)

    assert _CountingCodec.dump_calls == 1
    assert _CountingCodec.load_calls == 0
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest == {"$furu": {"kind": "lazy", "path": "lazy/root"}}
    assert (bundle_dir / "lazy" / "root" / "manifest.json").exists()

    loaded = load_result_bundle(bundle_dir)

    assert isinstance(loaded, LazyResult)
    assert not loaded.is_loaded
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


def test_nested_lazy_result_round_trips_inside_supported_structures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    _CountingCodec.load_calls = 0
    monkeypatch.setitem(_DEFAULT_CODECS, _CountingCodec.codec_id(), _CountingCodec)
    value = {
        "items": [
            LazyResult(_CountingValue(1)),
            {"inner": LazyResult((Path("x"), 2))},
        ]
    }

    save_result_bundle(value, bundle_dir)
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
        json.dumps({"$furu": {"kind": "lazy", "path": "../outside"}}),
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
        json.dumps({"$furu": {"kind": "lazy", "path": "lazy/root"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nested manifest missing"):
        load_result_bundle(bundle_dir)
