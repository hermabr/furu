from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Self

import pytest
from pydantic import BaseModel, model_validator

import furu
import furu.execution as execution_module
from furu.locking import LockLostError
from furu.metadata import RunningMetadata
from furu.results import load_result_bundle, save_result_bundle
from furu.results.errors import ResultCodecError, ResultSerializationError


@dataclass(frozen=True)
class CountedBlob:
    value: str


class CountedBlobCodec:
    codec_id = "test.counted-blob.v1"
    dump_calls: ClassVar[int] = 0
    load_calls: ClassVar[int] = 0

    def dump(self, value: Any, ctx: furu.DumpContext) -> furu.JsonValue:
        type(self).dump_calls += 1
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifact_dir / "data.txt").write_text(value.value, encoding="utf-8")
        return {"length": len(value.value)}

    def load(
        self,
        ctx: furu.LoadContext,
        meta: furu.JsonValue | None,
    ) -> CountedBlob:
        type(self).load_calls += 1
        return CountedBlob((ctx.artifact_dir / "data.txt").read_text(encoding="utf-8"))


@dataclass
class AnnotatedOutput:
    payload: Annotated[CountedBlob, furu.SaveWith(CountedBlobCodec.codec_id)]


@dataclass
class DataclassOutput:
    metrics: dict[str, float]
    weights: furu.LazyValue[CountedBlob]


@dataclass(frozen=True)
class DuckProtocolValue:
    name: str
    payload: bytes

    def __furu_result_dump__(self, ctx: furu.DumpContext) -> furu.JsonValue:
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifact_dir / "data.bin").write_bytes(self.payload)
        return {"name": self.name}

    @classmethod
    def __furu_result_load__(
        cls,
        ctx: furu.LoadContext,
        meta: furu.JsonValue,
    ) -> Self:
        assert isinstance(meta, dict)
        name = meta["name"]
        assert isinstance(name, str)
        return cls(
            name=name,
            payload=(ctx.artifact_dir / "data.bin").read_bytes(),
        )


@dataclass(frozen=True)
class MixinProtocolValue(furu.FuruResult):
    name: str
    payload: bytes

    def __furu_result_dump__(self, ctx: furu.DumpContext) -> furu.JsonValue:
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifact_dir / "data.bin").write_bytes(self.payload)
        return {"name": self.name}

    @classmethod
    def __furu_result_load__(
        cls,
        ctx: furu.LoadContext,
        meta: furu.JsonValue,
    ) -> Self:
        assert isinstance(meta, dict)
        name = meta["name"]
        assert isinstance(name, str)
        return cls(
            name=name,
            payload=(ctx.artifact_dir / "data.bin").read_bytes(),
        )


class OutputModel(BaseModel):
    metrics: dict[str, float]
    weights: object


class ValidationBypassModel(BaseModel):
    metrics: dict[str, float]
    weights: object
    validation_calls: ClassVar[int] = 0

    @model_validator(mode="after")
    def _reject_lazy_weights(self) -> Self:
        type(self).validation_calls += 1
        if isinstance(self.weights, furu.LazyValue):
            raise ValueError("normal validation should not run during result load")
        return self


class JsonOnlyResult(furu.Furu[dict[str, float]]):
    create_calls: ClassVar[int] = 0

    def _create(self) -> dict[str, float]:
        type(self).create_calls += 1
        return {"loss": 0.12, "accuracy": 0.94}


class LazyByPathResult(furu.Furu[dict[str, object]]):
    def _result_config(self) -> furu.ResultConfig:
        return furu.ResultConfig(rules=(furu.at("weights").lazy(),))

    def _create(self) -> dict[str, object]:
        return {
            "metrics": {"loss": 0.12},
            "weights": {"huge": [1, 2, 3]},
        }


class TypeCodecResult(furu.Furu[dict[str, object]]):
    def _result_config(self) -> furu.ResultConfig:
        registry = furu.ResultRegistry.default().with_codec(
            CountedBlobCodec(),
            types=(CountedBlob,),
        )
        return furu.ResultConfig(registry=registry)

    def _create(self) -> dict[str, object]:
        return {"x": CountedBlob("typed")}


class AnnotatedCodecResult(furu.Furu[AnnotatedOutput]):
    def _result_config(self) -> furu.ResultConfig:
        registry = furu.ResultRegistry.default().with_codec(CountedBlobCodec())
        return furu.ResultConfig(registry=registry)

    def _create(self) -> AnnotatedOutput:
        return AnnotatedOutput(payload=CountedBlob("annotated"))


class WrapperBeatsPathRuleResult(furu.Furu[dict[str, object]]):
    def _result_config(self) -> furu.ResultConfig:
        return furu.ResultConfig(rules=(furu.at("weights").lazy(),))

    def _create(self) -> dict[str, object]:
        return {
            "weights": furu.save_with(
                {"inline": True},
                codec="furu.json-tree.v1",
                lazy=False,
            )
        }


class PathBeatsTypeRegistryResult(furu.Furu[dict[str, object]]):
    def _result_config(self) -> furu.ResultConfig:
        registry = furu.ResultRegistry.default().with_codec(
            CountedBlobCodec(),
            types=(CountedBlob,),
        )
        return furu.ResultConfig(
            registry=registry,
            rules=(furu.at("weights").lazy(codec=CountedBlobCodec.codec_id),),
        )

    def _create(self) -> dict[str, object]:
        return {"weights": CountedBlob("path-rule")}


class DataclassResult(furu.Furu[DataclassOutput]):
    def _result_config(self) -> furu.ResultConfig:
        registry = furu.ResultRegistry.default().with_codec(CountedBlobCodec())
        return furu.ResultConfig(registry=registry)

    def _create(self) -> DataclassOutput:
        return DataclassOutput(
            metrics={"loss": 0.12},
            weights=furu.lazy(
                CountedBlob("dataclass"),
                codec=CountedBlobCodec.codec_id,
            ),
        )


class PydanticResult(furu.Furu[OutputModel]):
    def _result_config(self) -> furu.ResultConfig:
        registry = furu.ResultRegistry.default().with_codec(CountedBlobCodec())
        return furu.ResultConfig(
            registry=registry,
            rules=(furu.at("weights").lazy(codec=CountedBlobCodec.codec_id),),
        )

    def _create(self) -> OutputModel:
        return OutputModel(
            metrics={"loss": 0.12},
            weights=CountedBlob("pydantic"),
        )


class ValidationBypassResult(furu.Furu[ValidationBypassModel]):
    def _result_config(self) -> furu.ResultConfig:
        registry = furu.ResultRegistry.default().with_codec(CountedBlobCodec())
        return furu.ResultConfig(
            registry=registry,
            rules=(furu.at("weights").lazy(codec=CountedBlobCodec.codec_id),),
        )

    def _create(self) -> ValidationBypassModel:
        return ValidationBypassModel(
            metrics={"loss": 0.12},
            weights=CountedBlob("validated"),
        )


class PartialBatchResult(furu.Furu[str]):
    key: int

    @classmethod
    def _create_batched(cls, objs) -> list[str]:
        return [f"partial:{obj.key}" for obj in objs]


@pytest.fixture(autouse=True)
def _reset_counters() -> None:
    CountedBlobCodec.dump_calls = 0
    CountedBlobCodec.load_calls = 0
    JsonOnlyResult.create_calls = 0
    ValidationBypassModel.validation_calls = 0


def _roundtrip(
    tmp_path: Path,
    value: Any,
    *,
    config: furu.ResultConfig | None = None,
) -> tuple[Any, Path]:
    result_dir = tmp_path / "result"
    active_config = config or furu.ResultConfig()
    save_result_bundle(value, result_dir, active_config)
    return load_result_bundle(result_dir, active_config), result_dir


def _read_manifest(result_dir: Path) -> dict[str, Any]:
    return json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))


def test_json_only_result_creates_manifest_and_not_result_pkl() -> None:
    obj = JsonOnlyResult()

    assert obj.load_or_create() == {"loss": 0.12, "accuracy": 0.94}

    assert obj._result_manifest_path.exists()
    assert not (obj.data_dir / "result.pkl").exists()


def test_json_only_result_round_trips_exactly() -> None:
    obj = JsonOnlyResult()

    assert obj.load_or_create() == {"loss": 0.12, "accuracy": 0.94}
    assert obj.try_load() == {"loss": 0.12, "accuracy": 0.94}


def test_is_completed_checks_manifest_json() -> None:
    obj = JsonOnlyResult()

    assert not obj.is_completed()
    obj._result_dir.mkdir(parents=True)
    assert not obj.is_completed()
    save_result_bundle({"loss": 0.12}, obj._result_dir, obj._result_config())
    assert obj.is_completed()


def test_cache_hit_loads_from_result_bundle() -> None:
    obj = JsonOnlyResult()

    assert obj.load_or_create() == {"loss": 0.12, "accuracy": 0.94}
    assert obj.load_or_create() == {"loss": 0.12, "accuracy": 0.94}
    assert JsonOnlyResult.create_calls == 1


def test_fresh_compute_returns_loaded_representation_not_raw_create_output() -> None:
    out = LazyByPathResult().load_or_create()

    assert isinstance(out["weights"], furu.LazyValue)
    assert out["weights"].load() == {"huge": [1, 2, 3]}


def test_lazy_value_returns_lazy_value_on_first_run_and_cache_hit() -> None:
    first = LazyByPathResult().load_or_create()
    second = LazyByPathResult().load_or_create()

    assert isinstance(first["weights"], furu.LazyValue)
    assert isinstance(second["weights"], furu.LazyValue)
    assert first["weights"].load() == {"huge": [1, 2, 3]}
    assert second["weights"].load() == {"huge": [1, 2, 3]}


def test_lazy_load_cache_true_caches_materialized_value(tmp_path: Path) -> None:
    registry = furu.ResultRegistry.default().with_codec(CountedBlobCodec())
    value, _ = _roundtrip(
        tmp_path,
        furu.lazy(CountedBlob("cache"), codec=CountedBlobCodec.codec_id),
        config=furu.ResultConfig(registry=registry),
    )

    assert isinstance(value, furu.LazyValue)
    assert value.load(cache=True) == CountedBlob("cache")
    assert value.load(cache=True) == CountedBlob("cache")
    assert CountedBlobCodec.load_calls == 1


def test_lazy_load_cache_false_reloads_each_time(tmp_path: Path) -> None:
    registry = furu.ResultRegistry.default().with_codec(CountedBlobCodec())
    value, _ = _roundtrip(
        tmp_path,
        furu.lazy(CountedBlob("reload"), codec=CountedBlobCodec.codec_id),
        config=furu.ResultConfig(registry=registry),
    )

    assert isinstance(value, furu.LazyValue)
    assert value.load(cache=False) == CountedBlob("reload")
    assert value.load(cache=False) == CountedBlob("reload")
    assert CountedBlobCodec.load_calls == 2


def test_numpy_array_saves_as_npy_when_numpy_is_installed(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    value, result_dir = _roundtrip(
        tmp_path,
        {"weights": np.arange(10, dtype=np.int64)},
    )

    assert value["weights"].tolist() == list(range(10))
    assert (result_dir / "artifacts" / "weights" / "data.npy").exists()


def test_numpy_object_dtype_arrays_are_rejected_by_default(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    with pytest.raises(ResultCodecError, match="rejects object-dtype arrays"):
        save_result_bundle(
            {"weights": np.array([{"x": 1}], dtype=object)},
            tmp_path / "result",
            furu.ResultConfig(),
        )


def test_polars_dataframe_saves_as_parquet_when_polars_is_installed(
    tmp_path: Path,
) -> None:
    pl = pytest.importorskip("polars")

    value, result_dir = _roundtrip(
        tmp_path,
        {"table": pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})},
    )

    assert value["table"].to_dict(as_series=False) == {"x": [1, 2], "y": ["a", "b"]}
    assert (result_dir / "artifacts" / "table" / "data.parquet").exists()


def test_dataclass_result_round_trips_original_dataclass_type() -> None:
    out = DataclassResult().load_or_create()

    assert isinstance(out, DataclassOutput)
    assert isinstance(out.weights, furu.LazyValue)
    assert out.weights.load() == CountedBlob("dataclass")


def test_pydantic_result_round_trips_original_model_type() -> None:
    out = PydanticResult().load_or_create()

    assert isinstance(out, OutputModel)
    assert isinstance(out.weights, furu.LazyValue)
    assert out.weights.load() == CountedBlob("pydantic")


def test_pydantic_load_does_not_call_normal_validation() -> None:
    out = ValidationBypassResult().load_or_create()

    assert isinstance(out, ValidationBypassModel)
    assert isinstance(out.weights, furu.LazyValue)
    assert ValidationBypassModel.validation_calls == 1


def test_annotated_save_with_chooses_requested_codec() -> None:
    obj = AnnotatedCodecResult()

    out = obj.load_or_create()
    manifest = _read_manifest(obj._result_dir)

    assert out == AnnotatedOutput(payload=CountedBlob("annotated"))
    payload_node = manifest["root"]["$furu"]["fields"]["payload"]["$furu"]
    assert payload_node["codec"] == CountedBlobCodec.codec_id


def test_per_value_wrapper_beats_path_rule() -> None:
    out = WrapperBeatsPathRuleResult().load_or_create()

    assert out["weights"] == {"inline": True}
    assert not isinstance(out["weights"], furu.LazyValue)


def test_path_rule_beats_type_registry() -> None:
    out = PathBeatsTypeRegistryResult().load_or_create()

    assert isinstance(out["weights"], furu.LazyValue)
    assert out["weights"].load() == CountedBlob("path-rule")


def test_custom_codec_by_type_round_trips_registered_type() -> None:
    out = TypeCodecResult().load_or_create()

    assert out == {"x": CountedBlob("typed")}


def test_object_protocol_works_without_inheriting_furu_result(tmp_path: Path) -> None:
    value, result_dir = _roundtrip(tmp_path, DuckProtocolValue("duck", b"abc"))

    assert value == DuckProtocolValue("duck", b"abc")
    assert (result_dir / "artifacts" / "__root__" / "data.bin").exists()


def test_furu_result_mixin_works(tmp_path: Path) -> None:
    value, result_dir = _roundtrip(tmp_path, MixinProtocolValue("mixin", b"xyz"))

    assert value == MixinProtocolValue("mixin", b"xyz")
    assert (result_dir / "artifacts" / "__root__" / "data.bin").exists()


def test_mapping_with_reserved_furu_key_round_trips(tmp_path: Path) -> None:
    payload = {"$furu": "reserved", "other": 1}

    value, result_dir = _roundtrip(tmp_path, payload)
    manifest = _read_manifest(result_dir)

    assert value == payload
    assert manifest["root"]["$furu"]["kind"] == "mapping"


def test_mapping_with_non_string_keys_round_trips(tmp_path: Path) -> None:
    payload = {1: "one", (2, 3): "tuple"}

    value, result_dir = _roundtrip(tmp_path, payload)
    manifest = _read_manifest(result_dir)

    assert value == payload
    assert manifest["root"]["$furu"]["kind"] == "mapping"


def test_tuple_round_trips_as_tuple(tmp_path: Path) -> None:
    value, _ = _roundtrip(tmp_path, (1, "x", 3.5))

    assert value == (1, "x", 3.5)
    assert isinstance(value, tuple)


def test_set_and_frozenset_round_trip(tmp_path: Path) -> None:
    payload = {"set": {3, 1, 2}, "frozenset": frozenset({"a", "b"})}

    value, _ = _roundtrip(tmp_path, payload)

    assert value["set"] == {1, 2, 3}
    assert value["frozenset"] == frozenset({"a", "b"})
    assert isinstance(value["set"], set)
    assert isinstance(value["frozenset"], frozenset)


def test_cyclic_structures_raise_result_serialization_error(tmp_path: Path) -> None:
    payload: list[object] = []
    payload.append(payload)

    with pytest.raises(ResultSerializationError, match="cyclic result structure"):
        save_result_bundle(payload, tmp_path / "result", furu.ResultConfig())


def test_unsupported_type_error_includes_logical_path(tmp_path: Path) -> None:
    class UnsupportedValue:
        pass

    with pytest.raises(
        ResultSerializationError,
        match=r"result\.model\.weights",
    ):
        save_result_bundle(
            {"model": {"weights": UnsupportedValue()}},
            tmp_path / "result",
            furu.ResultConfig(),
        )


def test_lock_lost_before_final_rename_leaves_no_completed_manifest() -> None:
    obj = JsonOnlyResult()
    obj._internal_furu_dir.mkdir(parents=True, exist_ok=True)
    metadata = RunningMetadata.write_for(obj)
    has_lock_states = iter([True, False])

    def has_lock() -> bool:
        return next(has_lock_states)

    with pytest.raises(LockLostError, match="after writing temporary result"):
        execution_module._store_result(
            obj,
            {"loss": 0.12, "accuracy": 0.94},
            metadata=metadata,
            has_lock=has_lock,
        )

    assert not obj._result_manifest_path.exists()


def test_partial_persistence_keeps_already_stored_objects_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objs = [PartialBatchResult(key=1), PartialBatchResult(key=2)]
    real_store_result = execution_module._store_result
    call_count = 0

    def flaky_store_result(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("stop after first store")
        return real_store_result(*args, **kwargs)

    monkeypatch.setattr(execution_module, "_store_result", flaky_store_result)

    with pytest.raises(RuntimeError, match="stop after first store"):
        furu.load_or_create(objs)

    assert objs[0]._result_manifest_path.exists()
    assert not objs[1]._result_manifest_path.exists()
