from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from furu import Furu
from furu.config import config
from furu.result import (
    load_result_bundle,
    result_bundle_is_complete,
    save_result_bundle,
)

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]

try:
    import polars as _pl
except ImportError:  # pragma: no cover
    _pl = None  # type: ignore[assignment]


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


class ListResult(Furu[list[int]]):
    def _create(self) -> list[int]:
        return [1, 2, 3]


class _UnsupportedClass:
    def __init__(self, x: int = 0) -> None:
        self.x = x


class UnsupportedRoot(Furu[object]):
    def _create(self) -> object:
        return _UnsupportedClass()


class UnsupportedNested(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "layers": [
                {},
                {},
                {},
                {"weights": _UnsupportedClass()},
                {},
                {},
                {},
                {},
                {},
                {},
            ]
        }


class ReservedFuruKeyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"$furu": "user data"}


class NonStringKeyResult(Furu[dict]):
    def _create(self) -> dict:
        return {1: "x"}


class UnsafeKeyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"bad/key": "x"}


class EmptyKeyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"": "x"}


@dataclass(frozen=True)
class TrainOutput:
    metrics: dict[str, float]
    values: list[int]


class DataclassResult(Furu[TrainOutput]):
    def _create(self) -> TrainOutput:
        return TrainOutput(metrics={"loss": 0.12}, values=[1, 2, 3])


@dataclass(frozen=True)
class _NestedInner:
    label: str
    score: float


@dataclass(frozen=True)
class _NestedOuter:
    inner: _NestedInner
    items: list[_NestedInner]


class NestedDataclassResult(Furu[_NestedOuter]):
    def _create(self) -> _NestedOuter:
        return _NestedOuter(
            inner=_NestedInner(label="a", score=1.0),
            items=[
                _NestedInner(label="b", score=2.0),
                _NestedInner(label="c", score=3.0),
            ],
        )


class TrainOutputModel(BaseModel):
    metrics: dict[str, float]
    values: list[int]


class PydanticResult(Furu[TrainOutputModel]):
    def _create(self) -> TrainOutputModel:
        return TrainOutputModel(metrics={"loss": 0.12}, values=[1, 2, 3])


class _NestedInnerModel(BaseModel):
    label: str


class _NestedOuterModel(BaseModel):
    inner: _NestedInnerModel
    items: list[_NestedInnerModel]
    metrics: dict[str, float]


class NestedPydanticResult(Furu[_NestedOuterModel]):
    def _create(self) -> _NestedOuterModel:
        return _NestedOuterModel(
            inner=_NestedInnerModel(label="a"),
            items=[
                _NestedInnerModel(label="b"),
                _NestedInnerModel(label="c"),
            ],
            metrics={"loss": 0.12},
        )


# Module-level Furu classes for codec tests. They use `Any` typing and only
# touch numpy / polars at create time, so they can still be defined when those
# libraries are not installed (`_np` and `_pl` are simply `None` then; the
# tests that exercise them are gated by `pytest.importorskip`).


class NumpyResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"weights": _np.arange(10, dtype=_np.float32)}


class ObjectArrayResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"x": _np.array([object()], dtype=object)}


class RootArrayResult(Furu[Any]):
    def _create(self) -> Any:
        return _np.arange(4, dtype=_np.float64)


class NestedArraysResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "layers": [
                {"weights": _np.arange(i, dtype=_np.float32)} for i in range(10)
            ]
        }


class HundredLayersResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "layers": [
                {"weights": _np.arange(1, dtype=_np.float32)} for _ in range(100)
            ]
        }


class PolarsResult(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"table": _pl.DataFrame({"id": [1, 2], "score": [0.1, 0.2]})}


class ShortListResult(Furu[list[object]]):
    def _create(self) -> list[object]:
        return [_np.arange(1), _np.arange(2)]


class SingletonListResult(Furu[list[object]]):
    def _create(self) -> list[object]:
        return [_np.arange(3)]


# -----------------------------------------------------------------------------
# JSON-only result tests
# -----------------------------------------------------------------------------


def test_json_only_result_round_trip() -> None:
    obj = JsonResult()
    expected = {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }
    assert obj.load_or_create() == expected
    assert obj._result_manifest_path.exists()
    assert not (obj.data_dir / "result.pkl").exists()


def test_manifest_is_result_tree_directly_no_envelope() -> None:
    obj = JsonResult()
    obj.load_or_create()

    manifest = json.loads(obj._result_manifest_path.read_text())
    assert "format" not in manifest
    assert "root" not in manifest
    assert manifest["metrics"] == {"loss": 0.12, "ok": True}
    assert manifest["items"] == [1, 2, None, "x"]


def test_scalar_root_manifest_is_bare_value() -> None:
    obj = ScalarResult()
    assert obj.load_or_create() == 5

    manifest = json.loads(obj._result_manifest_path.read_text())
    assert manifest == 5


def test_list_root_manifest_is_bare_list() -> None:
    obj = ListResult()
    assert obj.load_or_create() == [1, 2, 3]
    manifest = json.loads(obj._result_manifest_path.read_text())
    assert manifest == [1, 2, 3]


def test_non_finite_floats_round_trip() -> None:
    obj = NonFiniteFloatResult()
    loaded = obj.load_or_create()

    assert math.isnan(loaded["nan"])
    assert loaded["pos_inf"] == float("inf")
    assert loaded["neg_inf"] == float("-inf")

    manifest_text = obj._result_manifest_path.read_text()
    # Python json writes NaN, Infinity, -Infinity in non-strict mode by default.
    assert "NaN" in manifest_text
    assert "Infinity" in manifest_text
    assert "-Infinity" in manifest_text


def test_cache_hit_does_not_recompute() -> None:
    JsonResult.create_calls = 0
    obj = JsonResult()
    expected = {
        "metrics": {"loss": 0.12, "ok": True},
        "items": [1, 2, None, "x"],
    }
    assert obj.load_or_create() == expected
    assert obj.load_or_create() == expected
    assert JsonResult.create_calls == 1


def test_is_completed_true_after_load_or_create() -> None:
    obj = JsonResult()
    assert not obj.is_completed()
    obj.load_or_create()
    assert obj.is_completed()


def test_try_load_returns_persisted_value() -> None:
    obj = JsonResult()
    expected = obj.load_or_create()
    assert obj.try_load() == expected


# -----------------------------------------------------------------------------
# Unsupported / error tests
# -----------------------------------------------------------------------------


def test_unsupported_root_value_includes_root_path_in_error() -> None:
    obj = UnsupportedRoot()
    with pytest.raises(ValueError) as excinfo:
        obj.load_or_create()
    msg = str(excinfo.value)
    assert "$" in msg
    assert "_UnsupportedClass" in msg


def test_unsupported_nested_value_includes_padded_index_path() -> None:
    obj = UnsupportedNested()
    with pytest.raises(ValueError) as excinfo:
        obj.load_or_create()
    msg = str(excinfo.value)
    assert "$.layers[03].weights" in msg


def test_reserved_furu_key_fails_clearly() -> None:
    obj = ReservedFuruKeyResult()
    with pytest.raises(ValueError) as excinfo:
        obj.load_or_create()
    msg = str(excinfo.value)
    assert "$furu" in msg
    assert "reserved" in msg


def test_non_string_dict_key_fails() -> None:
    obj = NonStringKeyResult()
    with pytest.raises(ValueError) as excinfo:
        obj.load_or_create()
    msg = str(excinfo.value)
    assert "must be strings" in msg


def test_unsafe_dict_key_fails() -> None:
    obj = UnsafeKeyResult()
    with pytest.raises(ValueError) as excinfo:
        obj.load_or_create()
    msg = str(excinfo.value)
    assert "bad/key" in msg
    assert "artifact path segment" in msg


def test_empty_dict_key_fails() -> None:
    obj = EmptyKeyResult()
    with pytest.raises(ValueError) as excinfo:
        obj.load_or_create()
    msg = str(excinfo.value)
    assert "artifact path segment" in msg


# -----------------------------------------------------------------------------
# Dataclass / Pydantic tests
# -----------------------------------------------------------------------------


def test_dataclass_result_round_trip() -> None:
    obj = DataclassResult()
    loaded = obj.load_or_create()
    assert isinstance(loaded, TrainOutput)
    assert loaded == TrainOutput(metrics={"loss": 0.12}, values=[1, 2, 3])


def test_nested_dataclass_round_trip() -> None:
    obj = NestedDataclassResult()
    loaded = obj.load_or_create()
    assert isinstance(loaded, _NestedOuter)
    assert loaded == _NestedOuter(
        inner=_NestedInner(label="a", score=1.0),
        items=[
            _NestedInner(label="b", score=2.0),
            _NestedInner(label="c", score=3.0),
        ],
    )


def test_dataclass_manifest_has_furu_wrapper() -> None:
    obj = DataclassResult()
    obj.load_or_create()
    manifest = json.loads(obj._result_manifest_path.read_text())
    assert "$furu" in manifest
    wrapper = manifest["$furu"]
    assert wrapper["kind"] == "dataclass"
    assert wrapper["type"].endswith("TrainOutput")
    assert wrapper["fields"]["metrics"] == {"loss": 0.12}
    assert wrapper["fields"]["values"] == [1, 2, 3]


def test_pydantic_result_round_trip() -> None:
    obj = PydanticResult()
    loaded = obj.load_or_create()
    assert isinstance(loaded, TrainOutputModel)
    assert loaded.metrics == {"loss": 0.12}
    assert loaded.values == [1, 2, 3]


def test_nested_pydantic_round_trip() -> None:
    obj = NestedPydanticResult()
    loaded = obj.load_or_create()
    assert isinstance(loaded, _NestedOuterModel)
    assert isinstance(loaded.inner, _NestedInnerModel)
    assert loaded.inner.label == "a"
    assert [item.label for item in loaded.items] == ["b", "c"]
    assert loaded.metrics == {"loss": 0.12}


def test_pydantic_manifest_has_furu_wrapper() -> None:
    obj = PydanticResult()
    obj.load_or_create()
    manifest = json.loads(obj._result_manifest_path.read_text())
    assert "$furu" in manifest
    wrapper = manifest["$furu"]
    assert wrapper["kind"] == "pydantic"
    assert wrapper["type"].endswith("TrainOutputModel")


# -----------------------------------------------------------------------------
# NumPy codec tests
# -----------------------------------------------------------------------------


def test_numpy_array_round_trip() -> None:
    np = pytest.importorskip("numpy")

    obj = NumpyResult()
    loaded = obj.load_or_create()

    artifact_path = obj._result_dir / "artifacts" / "weights" / "data.npy"
    assert artifact_path.exists()

    weights = loaded["weights"]
    assert weights.dtype == np.float32
    assert np.array_equal(weights, np.arange(10, dtype=np.float32))


def test_numpy_object_dtype_rejected() -> None:
    pytest.importorskip("numpy")

    obj = ObjectArrayResult()
    with pytest.raises(ValueError) as excinfo:
        obj.load_or_create()
    msg = str(excinfo.value)
    assert "object-dtype" in msg


def test_numpy_array_at_root_uses_root_artifact_path() -> None:
    np = pytest.importorskip("numpy")

    obj = RootArrayResult()
    loaded = obj.load_or_create()

    artifact_path = obj._result_dir / "artifacts" / "root" / "data.npy"
    assert artifact_path.exists()
    assert np.array_equal(loaded, np.arange(4, dtype=np.float64))

    manifest = json.loads(obj._result_manifest_path.read_text())
    assert "$furu" in manifest
    assert manifest["$furu"]["kind"] == "external"
    assert manifest["$furu"]["codec"] == "numpy.ndarray.npy"
    assert manifest["$furu"]["path"] == "artifacts/root"


def test_numpy_nested_artifact_path_uses_padded_indexes_for_length_10() -> None:
    pytest.importorskip("numpy")

    obj = NestedArraysResult()
    obj.load_or_create()

    base = obj._result_dir / "artifacts" / "layers"
    for i in range(10):
        path = base / f"{i:02d}" / "weights" / "data.npy"
        assert path.exists(), f"missing artifact at {path}"


def test_numpy_nested_artifact_path_uses_padded_indexes_for_length_100() -> None:
    pytest.importorskip("numpy")

    obj = HundredLayersResult()
    obj.load_or_create()

    path = obj._result_dir / "artifacts" / "layers" / "003" / "weights" / "data.npy"
    assert path.exists()


# -----------------------------------------------------------------------------
# Polars codec tests
# -----------------------------------------------------------------------------


def test_polars_dataframe_round_trip() -> None:
    pl = pytest.importorskip("polars")

    obj = PolarsResult()
    loaded = obj.load_or_create()

    artifact_path = obj._result_dir / "artifacts" / "table" / "data.parquet"
    assert artifact_path.exists()

    expected = pl.DataFrame({"id": [1, 2], "score": [0.1, 0.2]})
    assert loaded["table"].equals(expected)


# -----------------------------------------------------------------------------
# Path policy tests via internal API
# -----------------------------------------------------------------------------


def test_save_result_bundle_creates_manifest_and_no_pickle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    save_result_bundle({"a": 1}, bundle_dir)
    assert (bundle_dir / "manifest.json").exists()
    assert not (bundle_dir / "result.pkl").exists()
    assert result_bundle_is_complete(bundle_dir)


def test_save_result_bundle_fails_if_dir_exists(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    bundle_dir.mkdir()
    with pytest.raises(FileExistsError):
        save_result_bundle({"a": 1}, bundle_dir)


def test_load_result_bundle_round_trip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    save_result_bundle({"x": [1, 2, 3], "y": "ok"}, bundle_dir)
    assert load_result_bundle(bundle_dir) == {"x": [1, 2, 3], "y": "ok"}


def test_no_pickle_files_anywhere_in_data_dir() -> None:
    obj = JsonResult()
    obj.load_or_create()
    assert list(config.directories.data.glob("**/result.pkl")) == []


# -----------------------------------------------------------------------------
# List index padding policy tests via load_or_create
# -----------------------------------------------------------------------------


def test_list_index_padding_uses_list_length_width_for_short_list() -> None:
    pytest.importorskip("numpy")

    obj = ShortListResult()
    obj.load_or_create()
    base = obj._result_dir / "artifacts"
    # length 2, width = 1, indexes are "0" and "1"
    assert (base / "0" / "data.npy").exists()
    assert (base / "1" / "data.npy").exists()


def test_list_index_padding_for_length_one_list() -> None:
    pytest.importorskip("numpy")

    obj = SingletonListResult()
    obj.load_or_create()
    assert (obj._result_dir / "artifacts" / "0" / "data.npy").exists()
