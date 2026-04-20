from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import polars as pl
import pytest
from pydantic import BaseModel, ConfigDict

import furu
from furu import Furu, LazyValue, ResultConfig, SaveWith
from furu.results.bundle import load_result_bundle, save_result_bundle
from furu.results.nodes import DataclassNode, MappingNode, TupleNode
from furu.results.registry import ResultRegistry
from furu.results.rules import ResolveContext, resolve_plan
from furu.results.walker import DumpContext, dump_manifest
from furu.utils import JsonValue


class TextCodec:
    codec_id = "tests.text.v1"

    def dump(
        self,
        value: object,
        artifact_dir: Path,
        ctx: object,
    ) -> dict[str, JsonValue] | None:
        del ctx
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "data.txt").write_text(str(value), encoding="utf-8")
        return None

    def load(
        self,
        artifact_dir: Path,
        meta: dict[str, JsonValue] | None,
        ctx: object,
    ) -> object:
        del meta, ctx
        return (artifact_dir / "data.txt").read_text(encoding="utf-8")


@dataclass(frozen=True)
class ProtocolValue:
    text: str

    def __furu_result_dump__(self, ctx) -> object:
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifact_dir / "protocol.txt").write_text(self.text, encoding="utf-8")
        return ctx.external(codec="tests.protocol.v1")

    @classmethod
    def __furu_result_load__(cls, node, ctx) -> "ProtocolValue":
        del node
        return cls((ctx.artifact_dir / "protocol.txt").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class NestedMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class NestedOutput:
    metrics: NestedMetrics
    tags: tuple[str, ...]


@dataclass(frozen=True)
class AnnotatedOutput:
    metrics: dict[str, float]
    weights: Annotated[LazyValue[np.ndarray], SaveWith("numpy.ndarray.npy.v1")]


class PydanticOutput(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    table: Annotated[pl.DataFrame, SaveWith("polars.dataframe.parquet.v1")]
    weights: Annotated[LazyValue[np.ndarray], SaveWith("numpy.ndarray.npy.v1")]


class LazyTrainModel(Furu[dict[str, object]]):
    key: int

    def _result_config(self) -> ResultConfig:
        return ResultConfig(registry=furu.default_result_registry())

    def _create(self) -> dict[str, object]:
        return {
            "metrics": {"loss": 0.12},
            "weights": furu.lazy(
                np.arange(self.key + 2, dtype=np.float32),
                codec="numpy.ndarray.npy.v1",
            ),
        }


def _registry_with_text_codec() -> ResultRegistry:
    registry = furu.default_result_registry()
    registry.register_codec(TextCodec())
    registry.register_default_codec(ProtocolValue, "python.pickle.v1")
    return registry


def test_resolve_plan_explicit_wrapper_beats_other_sources() -> None:
    registry = _registry_with_text_codec()
    value = furu.save_with(ProtocolValue("x"), codec="tests.text.v1")

    raw_value, plan = resolve_plan(
        ResolveContext(
            value=value,
            logical_path=("value",),
            annotation=Annotated[ProtocolValue, SaveWith("python.pickle.v1")],
            registry=registry,
            rules=(furu.at("value").lazy(codec="python.pickle.v1"),),
        )
    )

    assert raw_value == ProtocolValue("x")
    assert plan.mode == "external"
    assert plan.codec_id == "tests.text.v1"
    assert not plan.lazy


def test_resolve_plan_path_rule_beats_annotation_protocol_and_registry() -> None:
    registry = _registry_with_text_codec()

    _, plan = resolve_plan(
        ResolveContext(
            value=ProtocolValue("x"),
            logical_path=("value",),
            annotation=Annotated[ProtocolValue, SaveWith("python.pickle.v1")],
            registry=registry,
            rules=(furu.at("value").save_as("tests.text.v1"),),
        )
    )

    assert plan.mode == "external"
    assert plan.codec_id == "tests.text.v1"


def test_resolve_plan_annotation_beats_protocol_and_registry() -> None:
    registry = _registry_with_text_codec()

    _, plan = resolve_plan(
        ResolveContext(
            value=ProtocolValue("x"),
            logical_path=(),
            annotation=Annotated[ProtocolValue, SaveWith("tests.text.v1")],
            registry=registry,
            rules=(),
        )
    )

    assert plan.mode == "external"
    assert plan.codec_id == "tests.text.v1"


def test_resolve_plan_prefers_protocol_over_registry() -> None:
    registry = _registry_with_text_codec()

    _, plan = resolve_plan(
        ResolveContext(
            value=ProtocolValue("x"),
            logical_path=(),
            annotation=None,
            registry=registry,
            rules=(),
        )
    )

    assert plan.mode == "structural"
    assert plan.protocol


def test_dump_manifest_detects_cycles() -> None:
    registry = furu.default_result_registry()
    value: list[object] = []
    value.append(value)

    with pytest.raises(ValueError, match="cycle detected"):
        dump_manifest(
            value,
            DumpContext(
                bundle_dir=Path("/unused"),
                artifacts_dir=Path("/unused/artifacts"),
                logical_path=(),
                registry=registry,
                rules=(),
            ),
        )


def test_dump_manifest_uses_structural_wrappers() -> None:
    registry = furu.default_result_registry()
    node = dump_manifest(
        {
            "data": NestedOutput(
                metrics=NestedMetrics(loss=0.1, accuracy=0.9),
                tags=("a", "b"),
            ),
            "mapping": {(1, 2): "value"},
        },
        DumpContext(
            bundle_dir=Path("/unused"),
            artifacts_dir=Path("/unused/artifacts"),
            logical_path=(),
            registry=registry,
            rules=(),
        ),
    )

    assert isinstance(node, dict)
    assert isinstance(node["data"], DataclassNode)
    assert isinstance(node["data"].fields["metrics"], DataclassNode)
    assert isinstance(node["data"].fields["tags"], TupleNode)
    assert isinstance(node["mapping"], MappingNode)


def test_json_only_bundle_uses_only_manifest(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = {"metrics": {"loss": 0.12}, "flags": [True, None, "ok"]}

    save_result_bundle(value, bundle_dir, config)

    assert [
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file()
    ] == ["manifest.json"]
    assert load_result_bundle(bundle_dir, config) == value


def test_dataclass_recursion_round_trips(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = NestedOutput(
        metrics=NestedMetrics(loss=0.1, accuracy=0.9),
        tags=("train", "eval"),
    )

    save_result_bundle(value, bundle_dir, config)
    loaded = load_result_bundle(bundle_dir, config)

    assert loaded == value
    assert isinstance(loaded, NestedOutput)


def test_dataclass_annotations_drive_lazy_round_trip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = AnnotatedOutput(
        metrics={"loss": 0.1},
        weights=cast(LazyValue[np.ndarray], np.arange(4, dtype=np.float32)),
    )

    save_result_bundle(value, bundle_dir, config)
    loaded = cast(AnnotatedOutput, load_result_bundle(bundle_dir, config))

    assert isinstance(loaded, AnnotatedOutput)
    assert isinstance(loaded.weights, LazyValue)
    assert not loaded.weights.is_loaded
    assert np.array_equal(loaded.weights.load(), np.arange(4, dtype=np.float32))


def test_pydantic_recursion_round_trips_without_model_dump(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    model = PydanticOutput.model_construct(
        table=df,
        weights=np.arange(3, dtype=np.float32),
    )

    save_result_bundle(model, bundle_dir, config)
    loaded = cast(PydanticOutput, load_result_bundle(bundle_dir, config))

    assert isinstance(loaded, PydanticOutput)
    assert loaded.table.equals(df)
    assert isinstance(loaded.weights, LazyValue)
    assert np.array_equal(loaded.weights.load(), np.arange(3, dtype=np.float32))


def test_numpy_codec_writes_npy_and_loads(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = np.arange(6, dtype=np.float32).reshape(2, 3)

    save_result_bundle(value, bundle_dir, config)
    loaded = cast(np.ndarray, load_result_bundle(bundle_dir, config))

    assert (bundle_dir / "artifacts" / "__root__" / "data.npy").exists()
    assert np.array_equal(loaded, value)


def test_numpy_codec_rejects_object_dtype_by_default(tmp_path: Path) -> None:
    config = ResultConfig(registry=furu.default_result_registry())

    with pytest.raises(TypeError, match="object-dtype"):
        save_result_bundle(
            np.array([{"x": 1}], dtype=object), tmp_path / "result", config
        )


def test_polars_codec_writes_parquet_and_loads(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    save_result_bundle(value, bundle_dir, config)
    loaded = cast(pl.DataFrame, load_result_bundle(bundle_dir, config))

    assert (bundle_dir / "artifacts" / "__root__" / "data.parquet").exists()
    assert loaded.equals(value)


def test_explicit_per_value_override_beats_type_default_registry(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = furu.save_with(np.arange(3, dtype=np.float32), codec="python.pickle.v1")

    save_result_bundle(value, bundle_dir, config)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    loaded = cast(np.ndarray, load_result_bundle(bundle_dir, config))

    assert manifest["root"]["$furu"]["codec"] == "python.pickle.v1"
    assert (bundle_dir / "artifacts" / "__root__" / "data.pkl").exists()
    assert np.array_equal(loaded, np.arange(3, dtype=np.float32))


def test_path_rule_can_override_one_json_subtree(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(
        registry=furu.default_result_registry(),
        rules=[furu.at("weights").lazy()],
    )
    value = {
        "weights": [1, 2, 3],
        "other": [4, 5],
    }

    save_result_bundle(value, bundle_dir, config)
    loaded = cast(dict[str, object], load_result_bundle(bundle_dir, config))

    assert isinstance(loaded["weights"], LazyValue)
    assert loaded["other"] == [4, 5]
    assert loaded["weights"].load() == [1, 2, 3]
    assert (bundle_dir / "artifacts" / "weights" / "data.json").exists()


def test_lazy_round_trip_and_caching(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = {
        "weights": furu.lazy(
            np.arange(5, dtype=np.float32),
            codec="numpy.ndarray.npy.v1",
        ),
    }

    save_result_bundle(value, bundle_dir, config)
    loaded = cast(dict[str, object], load_result_bundle(bundle_dir, config))
    weights = cast(LazyValue[np.ndarray], loaded["weights"])

    assert isinstance(weights, LazyValue)
    assert not weights.is_loaded
    first = weights.load()
    second = weights.load()
    assert first is second
    assert np.array_equal(first, np.arange(5, dtype=np.float32))


def test_object_protocol_round_trips_without_registered_codec(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = ProtocolValue("hello")

    save_result_bundle(value, bundle_dir, config)
    loaded = load_result_bundle(bundle_dir, config)

    assert loaded == value
    assert (bundle_dir / "artifacts" / "__root__" / "protocol.txt").exists()


def test_fresh_run_and_cache_hit_return_same_lazy_shape() -> None:
    obj = LazyTrainModel(key=3)

    first = obj.load_or_create()
    second = obj.load_or_create()

    first_weights = cast(LazyValue[np.ndarray], first["weights"])
    second_weights = cast(LazyValue[np.ndarray], second["weights"])

    assert isinstance(first_weights, LazyValue)
    assert isinstance(second_weights, LazyValue)
    assert not first_weights.is_loaded
    assert not second_weights.is_loaded
    assert first["metrics"] == second["metrics"] == {"loss": 0.12}
    assert np.array_equal(first_weights.load(), second_weights.load())


def test_artifact_paths_are_derived_from_logical_path(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "result"
    config = ResultConfig(registry=furu.default_result_registry())
    value = {
        "layers": [
            {
                "weights": furu.save_with(
                    np.arange(3, dtype=np.float32),
                    codec="numpy.ndarray.npy.v1",
                )
            }
        ]
    }

    save_result_bundle(value, bundle_dir, config)

    assert (
        bundle_dir / "artifacts" / "layers" / "000000" / "weights" / "data.npy"
    ).exists()


def test_unsupported_type_errors_are_path_aware(tmp_path: Path) -> None:
    class Unsupported:
        pass

    config = ResultConfig(registry=furu.default_result_registry())

    with pytest.raises(TypeError) as exc_info:
        save_result_bundle({"bad": Unsupported()}, tmp_path / "result", config)

    message = str(exc_info.value)
    assert "bad" in message
    assert "Unsupported" in message
    assert "Register a codec" in message


def test_load_result_bundle_fails_for_corrupt_bundles(tmp_path: Path) -> None:
    config = ResultConfig(registry=furu.default_result_registry())

    missing_dir = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_result_bundle(missing_dir, config)

    wrong_format_dir = tmp_path / "wrong-format"
    wrong_format_dir.mkdir()
    (wrong_format_dir / "manifest.json").write_text(
        json.dumps({"format": "wrong", "root": None}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported result manifest format"):
        load_result_bundle(wrong_format_dir, config)

    missing_artifact_dir = tmp_path / "missing-artifact"
    missing_artifact_dir.mkdir()
    (missing_artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "furu-result/v1",
                "root": {
                    "$furu": {
                        "kind": "external",
                        "codec": "json.file.v1",
                        "path": "artifacts/__root__",
                        "lazy": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="artifact directory"):
        load_result_bundle(missing_artifact_dir, config)
