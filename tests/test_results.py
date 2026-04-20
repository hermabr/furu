from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, ConfigDict

import furu
from furu import Furu
from furu.results import (
    DumpContext,
    FuruLazy,
    LoadContext,
    ResultConfig,
    default_result_config,
    default_result_registry,
    load_result_bundle,
    save_result_bundle,
)
from furu.results.errors import ResultCodecError, ResultSerializationError


@dataclass(frozen=True)
class MetricsLeaf:
    metrics: dict[str, float]
    pair: tuple[int, int]
    labels: set[int]
    frozen: frozenset[str]


@dataclass(frozen=True)
class MetricsTree:
    child: MetricsLeaf


@dataclass(frozen=True)
class FakeTensor:
    value: str


class DefaultFakeTensorCodec:
    codec_id = "tests.fake_tensor.default.v1"

    def dump(self, value: FakeTensor, ctx: DumpContext) -> dict[str, str]:
        (ctx.artifact_dir / "default.txt").write_text(value.value, encoding="utf-8")
        return {"codec": "default"}

    def load(self, ctx: LoadContext) -> FakeTensor:
        return FakeTensor(
            value=(ctx.artifact_dir / "default.txt").read_text(encoding="utf-8")
        )


class OverrideFakeTensorCodec:
    codec_id = "tests.fake_tensor.override.v1"

    def dump(self, value: FakeTensor, ctx: DumpContext) -> dict[str, str]:
        (ctx.artifact_dir / "override.txt").write_text(value.value, encoding="utf-8")
        return {"codec": "override"}

    def load(self, ctx: LoadContext) -> FakeTensor:
        return FakeTensor(
            value=(ctx.artifact_dir / "override.txt").read_text(encoding="utf-8")
        )


@dataclass(frozen=True)
class AnnotatedTensorResult:
    tensor: Annotated[FakeTensor, furu.SaveWith("tests.fake_tensor.override.v1")]


@dataclass(frozen=True)
class ProtocolValue:
    name: str

    def __furu_result_dump__(self, ctx: DumpContext) -> dict[str, str]:
        (ctx.artifact_dir / "data.txt").write_text(self.name, encoding="utf-8")
        return {"saved_name": self.name}

    @classmethod
    def __furu_result_load__(cls, payload: object, ctx: LoadContext) -> "ProtocolValue":
        assert isinstance(payload, dict)
        return cls(name=(ctx.artifact_dir / "data.txt").read_text(encoding="utf-8"))


class DumpGuardModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    label: str
    payload: object

    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("model_dump() should not be used by Furu result saving")


class UnsupportedValue:
    def __init__(self, name: str) -> None:
        self.name = name


@dataclass(frozen=True)
class FailingExternal:
    name: str


class FailingExternalCodec:
    codec_id = "tests.failing_external.v1"

    def dump(self, value: FailingExternal, ctx: DumpContext) -> None:
        raise ResultCodecError("forced dump failure")

    def load(self, ctx: LoadContext) -> FailingExternal:
        raise AssertionError("load() should not be called")


class LazyResultNode(Furu[dict[str, object]]):
    name: str

    def _create(self) -> dict[str, object]:
        return {"weights": furu.lazy({"values": [1, 2, 3]})}


class FailingResultNode(Furu[dict[str, object]]):
    name: str

    def _create(self) -> dict[str, object]:
        return {"bad": FailingExternal(self.name)}

    def _result_config(self) -> ResultConfig:
        registry = default_result_registry().with_type(
            FailingExternal, FailingExternalCodec()
        )
        return ResultConfig(registry=registry)


def _manifest(result_dir: Path) -> Any:
    return json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))


def test_json_only_result_bundle_has_manifest_without_artifacts(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"

    save_result_bundle(
        {"metrics": {"loss": 0.12}, "ok": True},
        result_dir,
        default_result_config(),
    )

    assert (result_dir / "manifest.json").exists()
    assert list((result_dir / "artifacts").iterdir()) == []
    assert load_result_bundle(result_dir, default_result_config()) == {
        "metrics": {"loss": 0.12},
        "ok": True,
    }


def test_reserved_key_mapping_round_trips(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    value = {"$furu": "user value", "|kind": "ordinary user key"}

    save_result_bundle(value, result_dir, default_result_config())

    manifest = _manifest(result_dir)
    assert manifest["root"] == {
        "$furu": {
            "kind": "mapping",
            "items": [
                {"key": "$furu", "value": "user value"},
                {"key": "|kind", "value": "ordinary user key"},
            ],
        }
    }
    assert load_result_bundle(result_dir, default_result_config()) == value


def test_dataclass_result_round_trips_recursively(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    value = MetricsTree(
        child=MetricsLeaf(
            metrics={"loss": 0.12},
            pair=(1, 2),
            labels={3, 1, 2},
            frozen=frozenset({"a", "b"}),
        )
    )

    save_result_bundle(value, result_dir, default_result_config())

    assert load_result_bundle(result_dir, default_result_config()) == value


def test_pydantic_result_round_trips_without_model_dump(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    value = DumpGuardModel(label="x", payload=furu.lazy({"items": [1, 2, 3]}))

    save_result_bundle(value, result_dir, default_result_config())

    loaded = load_result_bundle(result_dir, default_result_config())
    assert isinstance(loaded, DumpGuardModel)
    assert loaded.label == "x"
    assert isinstance(loaded.payload, FuruLazy)
    assert loaded.payload.load() == {"items": [1, 2, 3]}


def test_object_protocol_round_trips(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    value = ProtocolValue(name="checkpoint")

    save_result_bundle(value, result_dir, default_result_config())

    manifest = _manifest(result_dir)
    assert manifest["root"] == {
        "$furu": {
            "artifact_dir": "artifacts/root",
            "kind": "custom",
            "lazy": False,
            "payload": {"saved_name": "checkpoint"},
            "python_type": "test_results.ProtocolValue",
        }
    }
    assert load_result_bundle(result_dir, default_result_config()) == value


def test_numpy_codec_round_trips_with_nested_artifact_path(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    result_dir = tmp_path / "result"
    value = {"model": {"weights": np.arange(6, dtype=np.float32).reshape(2, 3)}}

    save_result_bundle(value, result_dir, default_result_config())

    manifest = _manifest(result_dir)
    weights_node = manifest["root"]["model"]["weights"]
    assert weights_node == {
        "$furu": {
            "artifact_dir": "artifacts/model/weights",
            "codec": "numpy.ndarray.npy.v1",
            "kind": "external",
            "lazy": False,
            "meta": {"dtype": "float32", "shape": [2, 3]},
            "python_type": "numpy.ndarray",
        }
    }
    assert (result_dir / "artifacts" / "model" / "weights" / "data.npy").exists()

    loaded = load_result_bundle(result_dir, default_result_config())
    assert np.array_equal(loaded["model"]["weights"], value["model"]["weights"])


def test_numpy_object_dtype_is_rejected(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    result_dir = tmp_path / "result"

    with pytest.raises(ResultSerializationError, match="object-dtype numpy arrays"):
        save_result_bundle(
            {"weights": np.array([object()], dtype=object)},
            result_dir,
            default_result_config(),
        )


def test_polars_codec_round_trips(tmp_path: Path) -> None:
    pl = pytest.importorskip("polars")
    result_dir = tmp_path / "result"
    value = pl.DataFrame({"id": [1, 2], "score": [0.1, 0.2]})

    save_result_bundle(value, result_dir, default_result_config())

    manifest = _manifest(result_dir)
    assert manifest["root"] == {
        "$furu": {
            "artifact_dir": "artifacts/root",
            "codec": "polars.dataframe.parquet.v1",
            "kind": "external",
            "lazy": False,
            "meta": {"columns": ["id", "score"], "height": 2},
            "python_type": "polars.dataframe.frame.DataFrame",
        }
    }
    assert (result_dir / "artifacts" / "root" / "data.parquet").exists()

    loaded = load_result_bundle(result_dir, default_result_config())
    assert loaded.equals(value)


def test_per_value_override_beats_registered_type_codec(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    registry = default_result_registry().with_type(FakeTensor, DefaultFakeTensorCodec())
    registry.register_codec(OverrideFakeTensorCodec())

    save_result_bundle(
        furu.result(FakeTensor("wrapped"), codec="tests.fake_tensor.override.v1"),
        result_dir,
        ResultConfig(registry=registry),
    )

    manifest = _manifest(result_dir)
    assert manifest["root"]["$furu"]["codec"] == "tests.fake_tensor.override.v1"
    assert (result_dir / "artifacts" / "root" / "override.txt").exists()
    assert load_result_bundle(
        result_dir, ResultConfig(registry=registry)
    ) == FakeTensor("wrapped")


def test_field_annotation_beats_type_registry(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    registry = default_result_registry().with_type(FakeTensor, DefaultFakeTensorCodec())
    registry.register_codec(OverrideFakeTensorCodec())

    save_result_bundle(
        AnnotatedTensorResult(tensor=FakeTensor("field")),
        result_dir,
        ResultConfig(registry=registry),
    )

    manifest = _manifest(result_dir)
    assert (
        manifest["root"]["$furu"]["fields"]["tensor"]["$furu"]["codec"]
        == "tests.fake_tensor.override.v1"
    )


def test_path_rule_can_make_value_lazy(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    config = ResultConfig(
        registry=default_result_registry(),
        rules=(furu.at("weights").lazy(),),
    )

    save_result_bundle({"weights": {"items": [1, 2, 3]}}, result_dir, config)

    loaded = load_result_bundle(result_dir, config)
    assert isinstance(loaded["weights"], FuruLazy)
    assert loaded["weights"].load() == {"items": [1, 2, 3]}


def test_lazy_first_run_matches_cache_hit_shape() -> None:
    node = LazyResultNode(name="x")

    first = node.load_or_create()
    second = node.load_or_create()

    assert isinstance(first["weights"], FuruLazy)
    assert isinstance(second["weights"], FuruLazy)
    assert first["weights"].load() == {"values": [1, 2, 3]}
    assert second["weights"].load() == {"values": [1, 2, 3]}


def test_lazy_load_caches_materialized_value(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"

    save_result_bundle(
        {"weights": furu.lazy({"nested": [1, 2]})},
        result_dir,
        default_result_config(),
    )

    loaded = load_result_bundle(result_dir, default_result_config())
    lazy_value = loaded["weights"]
    assert isinstance(lazy_value, FuruLazy)

    first = lazy_value.load()
    second = lazy_value.load()

    assert first is second


def test_unsupported_type_error_is_path_aware(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"

    with pytest.raises(ResultSerializationError) as exc_info:
        save_result_bundle(
            {"checkpoint": UnsupportedValue("x")},
            result_dir,
            default_result_config(),
        )

    message = str(exc_info.value)
    assert 'result["checkpoint"]' in message
    assert "test_results.UnsupportedValue" in message
    assert "furu.result(...)" in message
    assert "SaveWith(...)" in message


def test_atomic_failure_cleanup_does_not_leave_completed_bundle() -> None:
    node = FailingResultNode(name="boom")

    with pytest.raises(ResultSerializationError, match="forced dump failure"):
        node.load_or_create()

    assert not node._result_manifest_path.exists()


def test_no_new_results_use_pickle_paths() -> None:
    node = LazyResultNode(name="pickle-check")

    node.load_or_create()

    assert list(node.data_dir.glob("**/result.pkl")) == []
