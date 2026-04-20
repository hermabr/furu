from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import polars as pl
import pytest
from pydantic import BaseModel, ConfigDict

from furu import (
    Furu,
    FuruResult,
    LazyValue,
    ResultConfig,
    SaveWith,
    at,
    save_with,
    when_type,
)
from furu.results import load_result_bundle, save_result_bundle


@dataclass(frozen=True)
class ResultLeaf:
    name: str
    scores: tuple[int, ...]


@dataclass(frozen=True)
class ResultTree:
    leaf: ResultLeaf
    labels: set[str]
    tags: frozenset[str]


class ResultInnerModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    flags: tuple[bool, bool]


class ResultOuterModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    inner: ResultInnerModel


@dataclass(frozen=True)
class AnnotatedArrayResult:
    weights: Annotated[np.ndarray, SaveWith("pickle")]


class ProtocolResult(FuruResult):
    def __init__(self, *, left: int, right: int) -> None:
        self.left = left
        self.right = right

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ProtocolResult)
            and self.left == other.left
            and self.right == other.right
        )

    def __furu_save_result__(self, ctx: object) -> object:
        return {"left": self.left, "right": self.right}

    @classmethod
    def __furu_load_result__(cls, node: object, ctx: object) -> "ProtocolResult":
        payload = cast(dict[str, int], node)
        return cls(left=payload["left"], right=payload["right"])


class LazyWeightsNode(Furu[dict[str, object]]):
    key: int

    def _create(self) -> dict[str, object]:
        return {
            "weights": np.arange(self.key, self.key + 6, dtype=np.float32).reshape(
                2, 3
            ),
            "metrics": {"loss": 0.12},
        }

    def _result_config(self) -> ResultConfig:
        return ResultConfig(rules=(at("weights").lazy(),))


def _read_manifest(result_dir: Path) -> dict[str, object]:
    return json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))


def test_json_only_result_saves_manifest_only_and_round_trips(tmp_path) -> None:
    value = {
        "metrics": {"loss": 0.12, "accuracy": 0.94},
        "tags": ["baseline", "test"],
    }
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, ResultConfig.default())

    assert (result_dir / "manifest.json").exists()
    assert not (result_dir / "artifacts").exists()
    assert load_result_bundle(result_dir, ResultConfig.default()) == value

    manifest = _read_manifest(result_dir)
    assert manifest["format"] == "furu-result/v1"
    assert manifest["root"] == value


def test_dataclass_result_round_trips_recursively(tmp_path) -> None:
    value = ResultTree(
        leaf=ResultLeaf(name="leaf", scores=(1, 2, 3)),
        labels={"a", "b"},
        tags=frozenset({"x", "y"}),
    )
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, ResultConfig.default())
    loaded: ResultTree = load_result_bundle(result_dir, ResultConfig.default())

    assert loaded == value
    assert isinstance(loaded, ResultTree)


def test_pydantic_result_round_trips_recursively(tmp_path) -> None:
    value = ResultOuterModel(
        name="outer",
        inner=ResultInnerModel(count=3, flags=(True, False)),
    )
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, ResultConfig.default())
    loaded: ResultOuterModel = load_result_bundle(result_dir, ResultConfig.default())

    assert loaded == value
    assert isinstance(loaded, ResultOuterModel)


def test_numpy_default_codec_is_eager_and_writes_mirrored_artifact_tree(
    tmp_path,
) -> None:
    value = {
        "model": {"weights": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "metrics": {"loss": 0.12},
    }
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, ResultConfig.default())
    loaded: dict[str, object] = load_result_bundle(result_dir, ResultConfig.default())
    model = cast(dict[str, object], loaded["model"])
    loaded_weights = cast(np.ndarray, model["weights"])

    assert np.array_equal(loaded_weights, value["model"]["weights"])
    assert (result_dir / "artifacts" / "model" / "weights" / "data.npy").exists()

    manifest = _read_manifest(result_dir)
    weights = cast(
        dict[str, object], cast(dict[str, object], manifest["root"])["model"]
    )["weights"]
    external = cast(dict[str, object], cast(dict[str, object], weights)["$furu"])
    assert external["serializer"] == "numpy_npy"
    assert external["artifact_dir"] == "artifacts/model/weights"
    assert external["lazy"] is False
    assert external["meta"] == {"shape": [2, 3], "dtype": "float32"}


def test_polars_default_codec_is_eager_and_writes_parquet(tmp_path) -> None:
    value = {"frame": pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})}
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, ResultConfig.default())
    loaded: dict[str, object] = load_result_bundle(result_dir, ResultConfig.default())
    loaded_frame = cast(pl.DataFrame, loaded["frame"])

    assert loaded_frame.equals(value["frame"])
    assert (result_dir / "artifacts" / "frame" / "data.parquet").exists()

    manifest = _read_manifest(result_dir)
    frame = cast(dict[str, object], manifest["root"])["frame"]
    external = cast(dict[str, object], cast(dict[str, object], frame)["$furu"])
    assert external["serializer"] == "polars_parquet"
    assert external["lazy"] is False
    assert external["meta"] == {"shape": [2, 2], "columns": ["x", "y"]}


def test_per_value_override_beats_type_default(tmp_path) -> None:
    value = {"weights": save_with(np.arange(4, dtype=np.float32), serializer="pickle")}
    config = ResultConfig(rules=(when_type(np.ndarray).save_as("numpy_npy"),))
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, config)

    manifest = _read_manifest(result_dir)
    weights = cast(dict[str, object], manifest["root"])["weights"]
    external = cast(dict[str, object], cast(dict[str, object], weights)["$furu"])
    assert external["serializer"] == "pickle"


def test_path_rule_overrides_type_default(tmp_path) -> None:
    value = {"weights": np.arange(4, dtype=np.float32)}
    config = ResultConfig(
        rules=(
            when_type(np.ndarray).save_as("pickle"),
            at("weights").save_as("numpy_npy"),
        )
    )
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, config)

    manifest = _read_manifest(result_dir)
    weights = cast(dict[str, object], manifest["root"])["weights"]
    external = cast(dict[str, object], cast(dict[str, object], weights)["$furu"])
    assert external["serializer"] == "numpy_npy"


def test_field_annotation_overrides_type_default(tmp_path) -> None:
    value = AnnotatedArrayResult(weights=np.arange(4, dtype=np.float32))
    config = ResultConfig(rules=(when_type(np.ndarray).save_as("numpy_npy"),))
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, config)

    manifest = _read_manifest(result_dir)
    root = cast(dict[str, object], cast(dict[str, object], manifest["root"])["$furu"])
    weights = cast(
        dict[str, object], cast(dict[str, object], root["fields"])["weights"]
    )
    external = cast(dict[str, object], weights["$furu"])
    assert external["serializer"] == "pickle"


def test_lazy_round_trip_returns_lazy_value_and_loads_on_demand(tmp_path) -> None:
    value = {"weights": np.arange(6, dtype=np.float32).reshape(2, 3)}
    config = ResultConfig(rules=(at("weights").lazy(),))
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, config)
    loaded: dict[str, object] = load_result_bundle(result_dir, config)
    lazy_weights = cast(LazyValue[np.ndarray], loaded["weights"])

    assert isinstance(lazy_weights, LazyValue)
    assert np.array_equal(lazy_weights.load(), value["weights"])


def test_lazy_first_run_and_cache_hit_behave_the_same() -> None:
    obj = LazyWeightsNode(key=1)

    first: dict[str, object] = obj.load_or_create()
    second: dict[str, object] = obj.load_or_create()
    first_weights = cast(LazyValue[np.ndarray], first["weights"])
    second_weights = cast(LazyValue[np.ndarray], second["weights"])

    assert isinstance(first_weights, LazyValue)
    assert isinstance(second_weights, LazyValue)
    assert np.array_equal(first_weights.load(), second_weights.load())


def test_unsupported_type_error_includes_logical_path(tmp_path) -> None:
    with pytest.raises(TypeError, match=r'result\["bad"\]'):
        save_result_bundle(
            {"bad": object()}, tmp_path / "result", ResultConfig.default()
        )


def test_cycle_detection_fails_clearly(tmp_path) -> None:
    value: list[object] = []
    value.append(value)

    with pytest.raises(TypeError, match=r"Cycle detected while saving result\[0\]"):
        save_result_bundle(value, tmp_path / "result", ResultConfig.default())


def test_protocol_result_round_trips(tmp_path) -> None:
    value = {"point": ProtocolResult(left=3, right=5)}
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, ResultConfig.default())
    loaded: dict[str, object] = load_result_bundle(result_dir, ResultConfig.default())

    assert loaded == value


def test_json_file_serializer_round_trips_json_subtree(tmp_path) -> None:
    value = {
        "big_json": save_with({"nested": [1, 2, {"ok": True}]}, serializer="json_file")
    }
    result_dir = tmp_path / "result"

    save_result_bundle(value, result_dir, ResultConfig.default())
    loaded = load_result_bundle(result_dir, ResultConfig.default())

    assert loaded == {"big_json": {"nested": [1, 2, {"ok": True}]}}
    assert (result_dir / "artifacts" / "big_json" / "data.json").exists()
