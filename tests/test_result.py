from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import pytest
from pydantic import BaseModel, ConfigDict

from furu import Furu
from furu.result import (
    NumpyNpyCodec,
    load_result_bundle,
    save_result_bundle,
)


# ---------------------------------------------------------------------------
# JSON-only bundle test
# ---------------------------------------------------------------------------


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

    assert obj._result_manifest_path.exists()

    manifest = json.loads(obj._result_manifest_path.read_text())
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
    assert len(JsonResult.create_calls) == 1


def test_is_completed_flips_after_first_run() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    assert not obj.is_completed()
    obj.load_or_create()
    assert obj.is_completed()


def test_try_load_returns_persisted_result() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()

    obj.load_or_create()

    assert obj.try_load() == {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }


# ---------------------------------------------------------------------------
# Scalar root test
# ---------------------------------------------------------------------------


class ScalarResult(Furu[int]):
    def _create(self) -> int:
        return 5


def test_scalar_root_manifest_is_just_the_value() -> None:
    obj = ScalarResult()

    assert obj.load_or_create() == 5
    text = obj._result_manifest_path.read_text()
    assert json.loads(text) == 5


# ---------------------------------------------------------------------------
# Non-finite float test
# ---------------------------------------------------------------------------


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

    text = obj._result_manifest_path.read_text()
    # Python's standard library writes these as JSON extensions.
    assert "NaN" in text
    assert "Infinity" in text
    assert "-Infinity" in text


# ---------------------------------------------------------------------------
# Unsupported value tests
# ---------------------------------------------------------------------------


class _CustomTensor:
    pass


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


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------


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

    manifest = json.loads(obj._result_manifest_path.read_text())
    assert manifest["$furu"]["kind"] == "dataclass"
    assert manifest["$furu"]["type"] == "test_result.TrainOutput"
    assert manifest["$furu"]["fields"] == {
        "metrics": {"loss": 0.12},
        "values": [1, 2, 3],
    }


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


# ---------------------------------------------------------------------------
# Pydantic round-trip
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# NumPy codec
# ---------------------------------------------------------------------------


class NumpyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        np = pytest.importorskip("numpy")
        return {"weights": np.arange(10, dtype=np.float32)}


def test_numpy_array_round_trips() -> None:
    np = pytest.importorskip("numpy")
    obj = NumpyResult()
    loaded = obj.load_or_create()

    assert (obj._result_dir / "artifacts" / "weights" / "data.npy").exists()
    assert isinstance(loaded, dict)
    weights = cast(Any, loaded["weights"])
    assert weights.dtype == np.float32
    assert np.array_equal(weights, np.arange(10, dtype=np.float32))

    manifest = json.loads(obj._result_manifest_path.read_text())
    assert manifest["weights"]["$furu"]["kind"] == "external"
    assert manifest["weights"]["$furu"]["codec"] == (
        f"{NumpyNpyCodec.__module__}.{NumpyNpyCodec.__qualname__}"
    )
    assert manifest["weights"]["$furu"]["path"] == "artifacts/weights"


def test_numpy_object_dtype_is_rejected(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="allow_pickle=False"):
        save_result_bundle(
            {"weights": np.array([object()], dtype=object)},
            bundle_dir,
        )


class NestedNumpyResult(Furu[dict[str, list[dict[str, object]]]]):
    def _create(self) -> dict[str, list[dict[str, object]]]:
        np = pytest.importorskip("numpy")
        return {
            "layers": [{"weights": np.arange(i, dtype=np.float32)} for i in range(10)]
        }


def test_nested_numpy_paths_use_list_length_padded_indexes() -> None:
    np = pytest.importorskip("numpy")
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
    np = pytest.importorskip("numpy")
    bundle_dir = tmp_path / "bundle"
    layers = [{"weights": np.arange(0, dtype=np.float32)} for _ in range(100)]
    layers[3] = {"weights": np.arange(3, dtype=np.float32)}

    save_result_bundle({"layers": layers}, bundle_dir)

    expected = bundle_dir / "artifacts" / "layers" / "003" / "weights" / "data.npy"
    assert expected.exists()


def test_numpy_root_value_uses_root_artifact_dir(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    bundle_dir = tmp_path / "bundle"
    save_result_bundle(np.arange(5, dtype=np.int64), bundle_dir)

    assert (bundle_dir / "artifacts" / "root" / "data.npy").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["$furu"]["kind"] == "external"
    assert manifest["$furu"]["path"] == "artifacts/root"

    loaded = load_result_bundle(bundle_dir)
    assert np.array_equal(loaded, np.arange(5, dtype=np.int64))


# ---------------------------------------------------------------------------
# Mixed JSON + external + dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixedOutput:
    metrics: dict[str, float]
    values: list[int]


class MixedResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        np = pytest.importorskip("numpy")
        return {
            "result": MixedOutput(metrics={"loss": 0.5}, values=[1, 2]),
            "weights": np.arange(4, dtype=np.float32),
            "labels": ["cat", "dog"],
        }


def test_mixed_dataclass_external_and_json_round_trip() -> None:
    np = pytest.importorskip("numpy")
    obj = MixedResult()
    loaded = obj.load_or_create()

    assert isinstance(loaded, dict)
    inner = loaded["result"]
    assert isinstance(inner, MixedOutput)
    assert inner == MixedOutput(metrics={"loss": 0.5}, values=[1, 2])
    assert np.array_equal(loaded["weights"], np.arange(4, dtype=np.float32))
    assert loaded["labels"] == ["cat", "dog"]


# ---------------------------------------------------------------------------
# Bundle API edge cases
# ---------------------------------------------------------------------------


def test_save_result_bundle_refuses_existing_directory(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    with pytest.raises(FileExistsError):
        save_result_bundle({"x": 1}, bundle_dir)


def test_save_result_bundle_writes_manifest_last(tmp_path) -> None:
    np = pytest.importorskip("numpy")
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
