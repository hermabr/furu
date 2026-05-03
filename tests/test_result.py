from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import pytest
from pydantic import BaseModel

from furu import Furu
from furu.config import config


class JsonResult(Furu[dict[str, object]]):
    create_calls: ClassVar[int] = 0

    def _create(self) -> dict[str, object]:
        type(self).create_calls += 1
        return {
            "metrics": {"loss": 0.12, "ok": True},
            "items": [1, 2, None, "x"],
        }


class ScalarResult(Furu[int]):
    def _create(self) -> int:
        return 5


class NonFiniteFloatResult(Furu[dict[str, float]]):
    def _create(self) -> dict[str, float]:
        return {
            "nan": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
        }


class Unsupported:
    pass


class UnsupportedResult(Furu[object]):
    def _create(self) -> object:
        return Unsupported()


class UnsupportedNestedResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "layers": [
                {},
                {},
                {},
                {"weights": Unsupported()},
                {},
                {},
                {},
                {},
                {},
                {},
            ]
        }


class ReservedKeyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"$furu": "user data"}


class NonStringKeyResult(Furu[dict[object, object]]):
    def _create(self) -> dict[object, object]:
        return {1: "x"}


class UnsafeKeyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"bad/key": "x"}


class EmptyKeyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"": "x"}


@dataclass(frozen=True)
class Output:
    metrics: dict[str, float]
    values: list[int]


class DataclassResult(Furu[Output]):
    def _create(self) -> Output:
        return Output(metrics={"loss": 0.12}, values=[1, 2, 3])


class OutputModel(BaseModel):
    metrics: dict[str, float]
    values: list[int]


class PydanticResult(Furu[OutputModel]):
    def _create(self) -> OutputModel:
        return OutputModel(metrics={"loss": 0.12}, values=[1, 2, 3])


class NumpyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        np = pytest.importorskip("numpy")
        return {"weights": np.arange(10, dtype=np.float32)}


class NumpyObjectDtypeResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        np = pytest.importorskip("numpy")
        return {"weights": np.array([object()], dtype=object)}


class NestedNumpyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        np = pytest.importorskip("numpy")
        return {
            "layers": [
                {"weights": np.arange(0)},
                {"weights": np.arange(1)},
                {"weights": np.arange(2)},
                {"weights": np.arange(3)},
                {"weights": np.arange(4)},
                {"weights": np.arange(5)},
                {"weights": np.arange(6)},
                {"weights": np.arange(7)},
                {"weights": np.arange(8)},
                {"weights": np.arange(9)},
            ]
        }


class HundredItemNumpyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        np = pytest.importorskip("numpy")
        return {"layers": [{"weights": np.arange(index)} for index in range(100)]}


class PolarsResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        pl = pytest.importorskip("polars")
        return {"table": pl.DataFrame({"id": [1, 2], "score": [0.1, 0.2]})}


def test_json_only_bundle_manifest_is_result_tree() -> None:
    JsonResult.create_calls = 0
    obj = JsonResult()
    expected = {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }

    assert obj.load_or_create() == expected
    assert obj._result_manifest_path.exists()
    assert not (obj.data_dir / "result.pkl").exists()

    manifest = json.loads(obj._result_manifest_path.read_text(encoding="utf-8"))
    assert manifest == expected
    assert "format" not in manifest
    assert "root" not in manifest


def test_scalar_root_manifest_is_bare_scalar() -> None:
    obj = ScalarResult()

    assert obj.load_or_create() == 5
    assert json.loads(obj._result_manifest_path.read_text(encoding="utf-8")) == 5


def test_non_finite_float_result_round_trips_with_builtin_json_behavior() -> None:
    obj = NonFiniteFloatResult()

    result = obj.load_or_create()

    assert math.isnan(result["nan"])
    assert result["pos_inf"] == float("inf")
    assert result["neg_inf"] == float("-inf")
    manifest = json.loads(obj._result_manifest_path.read_text(encoding="utf-8"))
    assert math.isnan(manifest["nan"])
    assert manifest["pos_inf"] == float("inf")
    assert manifest["neg_inf"] == float("-inf")


def test_cache_hit_uses_persisted_result() -> None:
    JsonResult.create_calls = 0
    obj = JsonResult()
    expected = {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }

    assert obj.load_or_create() == expected
    assert obj.load_or_create() == expected
    assert JsonResult.create_calls == 1


def test_result_manifest_drives_completion_and_try_load() -> None:
    obj = JsonResult()
    expected = {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }

    assert not obj.is_completed()
    assert obj.load_or_create() == expected
    assert obj.is_completed()
    assert obj.try_load() == expected


def test_unsupported_custom_object_error_includes_root_path_and_type() -> None:
    with pytest.raises(ValueError) as exc_info:
        UnsupportedResult().load_or_create()

    message = str(exc_info.value)
    assert "$" in message
    assert "Unsupported" in message


def test_unsupported_nested_value_error_uses_padded_logical_path() -> None:
    with pytest.raises(ValueError) as exc_info:
        UnsupportedNestedResult().load_or_create()

    assert "$.layers[03].weights" in str(exc_info.value)


def test_reserved_furu_dict_key_fails_clearly() -> None:
    with pytest.raises(ValueError, match=r"keys named '\$furu' are reserved"):
        ReservedKeyResult().load_or_create()


def test_non_string_dict_key_fails_clearly() -> None:
    with pytest.raises(ValueError) as exc_info:
        NonStringKeyResult().load_or_create()

    message = str(exc_info.value)
    assert "dict result keys must be strings" in message
    assert "int key 1" in message


def test_unsafe_dict_key_fails_clearly() -> None:
    with pytest.raises(ValueError) as exc_info:
        UnsafeKeyResult().load_or_create()

    message = str(exc_info.value)
    assert '$["bad/key"]' in message
    assert "dict key cannot be used as an artifact path segment" in message


def test_empty_dict_key_fails_clearly() -> None:
    with pytest.raises(ValueError) as exc_info:
        EmptyKeyResult().load_or_create()

    message = str(exc_info.value)
    assert '$[""]' in message
    assert "dict key cannot be used as an artifact path segment" in message


def test_dataclass_result_round_trips_as_dataclass_instance() -> None:
    obj = DataclassResult()

    loaded = obj.load_or_create()

    assert isinstance(loaded, Output)
    assert loaded == Output(metrics={"loss": 0.12}, values=[1, 2, 3])


def test_pydantic_result_round_trips_as_pydantic_model_instance() -> None:
    obj = PydanticResult()

    loaded = obj.load_or_create()

    assert isinstance(loaded, OutputModel)
    assert loaded.metrics == {"loss": 0.12}
    assert loaded.values == [1, 2, 3]


def test_numpy_array_codec_round_trips_and_writes_npy_artifact() -> None:
    np = pytest.importorskip("numpy")
    obj = NumpyResult()
    expected = np.arange(10, dtype=np.float32)

    loaded = obj.load_or_create()
    weights = cast(Any, loaded["weights"])

    assert (obj._result_dir / "artifacts" / "weights" / "data.npy").exists()
    assert weights.dtype == np.float32
    assert np.array_equal(weights, expected)


def test_numpy_object_dtype_arrays_are_rejected() -> None:
    pytest.importorskip("numpy")

    with pytest.raises(ValueError) as exc_info:
        NumpyObjectDtypeResult().load_or_create()

    message = str(exc_info.value)
    assert "$.weights" in message
    assert "numpy object-dtype arrays are not supported" in message


def test_polars_dataframe_codec_round_trips_and_writes_parquet_artifact() -> None:
    pl = pytest.importorskip("polars")
    obj = PolarsResult()
    expected = pl.DataFrame({"id": [1, 2], "score": [0.1, 0.2]})

    loaded = obj.load_or_create()
    table = cast(Any, loaded["table"])

    assert (obj._result_dir / "artifacts" / "table" / "data.parquet").exists()
    assert table.equals(expected)


def test_nested_artifact_paths_use_list_length_padding() -> None:
    pytest.importorskip("numpy")
    obj = NestedNumpyResult()

    obj.load_or_create()

    for index in range(10):
        assert (
            obj._result_dir
            / "artifacts"
            / "layers"
            / f"{index:02d}"
            / "weights"
            / "data.npy"
        ).exists()


def test_nested_artifact_path_for_length_100_uses_three_digits() -> None:
    pytest.importorskip("numpy")
    obj = HundredItemNumpyResult()

    obj.load_or_create()

    assert (
        obj._result_dir / "artifacts" / "layers" / "003" / "weights" / "data.npy"
    ).exists()


def test_load_or_create_never_writes_result_pickle() -> None:
    assert ScalarResult().load_or_create() == 5

    assert list(config.directories.data.glob("**/result.pkl")) == []
