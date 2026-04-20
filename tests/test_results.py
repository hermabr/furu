from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from typing import Annotated, Any, cast

import pytest
from pydantic import BaseModel, ConfigDict

import furu
from furu import Furu
from furu.results import (
    DumpContext,
    LoadContext,
    ResultConfig,
    ResultSerializationError,
    UnknownResultCodecError,
    load_result_bundle,
)


def _manifest(obj: Furu[Any]) -> dict[str, Any]:
    return json.loads(obj._result_manifest_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class DataClassResult:
    metrics: dict[str, float]
    values: list[int]


@dataclass(frozen=True)
class AnnotatedListResult:
    values: Annotated[list[int], furu.SaveWith("furu.json.v1")]


class PydanticResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float]
    values: list[int]


@dataclass(frozen=True)
class CustomValue:
    text: str


@dataclass(frozen=True)
class TypeRuleValue:
    text: str


@dataclass(frozen=True)
class FailingCodecValue:
    text: str


class CustomValueCodec:
    codec_id = "test.custom.v1"

    def dump(self, value: CustomValue, ctx: DumpContext):
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifact_dir / "value.txt").write_text(value.text, encoding="utf-8")
        return {"length": len(value.text)}

    def load(self, ctx: LoadContext, meta):
        del meta
        return CustomValue((ctx.artifact_dir / "value.txt").read_text(encoding="utf-8"))


class TypeRuleValueCodec:
    codec_id = "test.type-rule.v1"

    def dump(self, value: TypeRuleValue, ctx: DumpContext):
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifact_dir / "value.txt").write_text(value.text, encoding="utf-8")
        return {"source": "type-rule"}

    def load(self, ctx: LoadContext, meta):
        del meta
        return TypeRuleValue(
            (ctx.artifact_dir / "value.txt").read_text(encoding="utf-8")
        )


class FailingCodec:
    codec_id = "test.fail-after-write.v1"

    def dump(self, value: FailingCodecValue, ctx: DumpContext):
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifact_dir / "value.txt").write_text(value.text, encoding="utf-8")
        raise RuntimeError("codec write failed")

    def load(self, ctx: LoadContext, meta):
        del ctx, meta
        raise AssertionError("load should never be called")


class ProtocolValue(furu.FuruResult):
    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProtocolValue) and other.text == self.text

    def __furu_result_dump__(self, ctx: DumpContext):
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        (ctx.artifact_dir / "value.txt").write_text(self.text, encoding="utf-8")
        return {"format": 1}

    @classmethod
    def __furu_result_load__(cls, ctx: LoadContext, meta):
        assert meta == {"format": 1}
        return cls((ctx.artifact_dir / "value.txt").read_text(encoding="utf-8"))


class UnsupportedValue:
    pass


class JsonBundleProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"nested": {"ok": True}, "numbers": [1, 2, 3], "value": "done"}


class NaNProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"value": float("nan")}


class InfProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"value": float("inf")}


class TupleSetProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "frozen": frozenset({"x", "y"}),
            "set": {"a", "b"},
            "tuple": (1, 2, 3),
        }


class ReservedKeyProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"$furu": "user value", "x": 1}


class DataClassProbe(Furu[DataClassResult]):
    def _create(self) -> DataClassResult:
        return DataClassResult(metrics={"accuracy": 0.91}, values=[1, 2, 3])


class PydanticProbe(Furu[PydanticResult]):
    def _create(self) -> PydanticResult:
        return PydanticResult(metrics={"accuracy": 0.92}, values=[3, 4, 5])


class ExplicitExternalJsonProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"large": furu.result([1, 2, 3], codec="furu.json.v1")}


class LazyJsonProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"large": furu.lazy([1, 2, 3])}


class AnnotationProbe(Furu[AnnotatedListResult]):
    def _create(self) -> AnnotatedListResult:
        return AnnotatedListResult(values=[1, 2, 3])


class PathRuleLazyProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"payload": [1, 2, 3]}

    def _result_config(self) -> ResultConfig:
        config = ResultConfig.default()
        config.rules.append(furu.result_at("payload").save_with("furu.json.v1").lazy())
        return config


class CustomCodecProbe(Furu[CustomValue]):
    def _create(self) -> CustomValue:
        return CustomValue("hello")

    def _result_config(self) -> ResultConfig:
        config = ResultConfig.default()
        config.registry.register_type(CustomValue, CustomValueCodec())
        return config


class TypeRuleProbe(Furu[TypeRuleValue]):
    def _create(self) -> TypeRuleValue:
        return TypeRuleValue("rule-based")

    def _result_config(self) -> ResultConfig:
        config = ResultConfig.default()
        config.registry.register_codec(TypeRuleValueCodec())
        config.rules.append(
            furu.result_when_type(TypeRuleValue).save_with("test.type-rule.v1")
        )
        return config


class ProtocolProbe(Furu[ProtocolValue]):
    def _create(self) -> ProtocolValue:
        return ProtocolValue("protocol")


class NumpyProbe(Furu[object]):
    lazy_mode: bool = False

    def _create(self) -> object:
        np = import_module("numpy")

        value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        if self.lazy_mode:
            return furu.lazy(value)
        return value


class NumpyObjectProbe(Furu[object]):
    def _create(self) -> object:
        np = import_module("numpy")

        return np.array([{"x": 1}], dtype=object)


class PolarsProbe(Furu[object]):
    lazy_mode: bool = False

    def _create(self) -> object:
        pl = import_module("polars")

        value = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        if self.lazy_mode:
            return furu.lazy(value)
        return value


class NestedLazyPathProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "model": {
                "layers": [
                    furu.lazy([1]),
                    furu.lazy([2]),
                ]
            }
        }


class UnsupportedProbe(Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {"bad": UnsupportedValue()}


class CycleProbe(Furu[list[object]]):
    def _create(self) -> list[object]:
        values: list[object] = []
        values.append(values)
        return values


class FailingSaveProbe(Furu[FailingCodecValue]):
    def _create(self) -> FailingCodecValue:
        return FailingCodecValue("boom")

    def _result_config(self) -> ResultConfig:
        config = ResultConfig.default()
        config.registry.register_type(FailingCodecValue, FailingCodec())
        return config


def test_json_bundle_basics() -> None:
    obj = JsonBundleProbe()

    assert obj.load_or_create() == {
        "nested": {"ok": True},
        "numbers": [1, 2, 3],
        "value": "done",
    }
    assert obj._result_manifest_path.exists()
    assert not (obj.data_dir / "result.pkl").exists()

    manifest = _manifest(obj)
    assert list(manifest) == ["format", "root", "version"] or set(manifest) == {
        "format",
        "root",
        "version",
    }
    assert manifest["format"] == "furu-result-bundle"
    assert manifest["version"] == 1
    assert not (obj._result_dir / "artifacts").exists()


@pytest.mark.parametrize("probe_cls", [NaNProbe, InfProbe])
def test_strict_json_rejects_non_finite_floats(
    probe_cls: type[Furu[dict[str, object]]],
) -> None:
    with pytest.raises(ResultSerializationError, match=r'result\["value"\]'):
        probe_cls().load_or_create()


def test_tuple_set_and_frozenset_round_trip() -> None:
    obj = TupleSetProbe()

    result = obj.load_or_create()
    assert result["tuple"] == (1, 2, 3)
    assert result["set"] == {"a", "b"}
    assert result["frozen"] == frozenset({"x", "y"})

    root = _manifest(obj)["root"]
    assert root["tuple"]["$furu"]["kind"] == "tuple"
    assert root["set"]["$furu"]["kind"] == "set"
    assert root["frozen"]["$furu"]["kind"] == "frozenset"


def test_reserved_key_uses_mapping_escape_node() -> None:
    obj = ReservedKeyProbe()

    assert obj.load_or_create() == {"$furu": "user value", "x": 1}
    assert _manifest(obj)["root"]["$furu"]["kind"] == "mapping"


def test_dataclass_result_round_trips() -> None:
    obj = DataClassProbe()

    result = obj.load_or_create()
    assert isinstance(result, DataClassResult)
    assert result == DataClassResult(metrics={"accuracy": 0.91}, values=[1, 2, 3])

    root = _manifest(obj)["root"]["$furu"]
    assert root["kind"] == "dataclass"
    assert root["python_type"] == "test_results.DataClassResult"


def test_pydantic_result_round_trips() -> None:
    obj = PydanticProbe()

    result = obj.load_or_create()
    assert isinstance(result, PydanticResult)
    assert result.metrics["accuracy"] == 0.92
    assert result.values == [3, 4, 5]

    root = _manifest(obj)["root"]["$furu"]
    assert root["kind"] == "pydantic"
    assert root["python_type"] == "test_results.PydanticResult"


def test_json_file_codec_externalizes_explicit_value() -> None:
    obj = ExplicitExternalJsonProbe()

    assert obj.load_or_create() == {"large": [1, 2, 3]}
    root = _manifest(obj)["root"]["large"]["$furu"]
    assert root["kind"] == "external"
    assert root["codec"] == "furu.json.v1"
    assert (obj._result_dir / "artifacts" / "large" / "value.json").exists()


def test_lazy_json_value_round_trips_as_lazy_value() -> None:
    obj = LazyJsonProbe()

    first = obj.load_or_create()
    assert isinstance(first["large"], furu.LazyValue)
    assert not first["large"].is_loaded
    assert first["large"].load() == [1, 2, 3]
    assert first["large"].is_loaded
    assert first["large"].load() is first["large"].load()

    second = obj.load_or_create()
    assert isinstance(second["large"], furu.LazyValue)
    assert not second["large"].is_loaded
    assert second["large"].load() == [1, 2, 3]


def test_annotation_override_externalizes_dataclass_field() -> None:
    obj = AnnotationProbe()

    result = obj.load_or_create()
    assert result == AnnotatedListResult(values=[1, 2, 3])
    root = _manifest(obj)["root"]["$furu"]["fields"]["values"]["$furu"]
    assert root["codec"] == "furu.json.v1"
    assert (obj._result_dir / "artifacts" / "values" / "value.json").exists()


def test_path_rule_override_loads_lazy_value() -> None:
    obj = PathRuleLazyProbe()

    result = obj.load_or_create()
    assert isinstance(result["payload"], furu.LazyValue)
    assert result["payload"].load() == [1, 2, 3]


def test_custom_codec_round_trips_and_unknown_codec_is_helpful() -> None:
    obj = CustomCodecProbe()

    result = obj.load_or_create()
    assert result == CustomValue("hello")
    assert _manifest(obj)["root"]["$furu"]["codec"] == "test.custom.v1"

    with pytest.raises(UnknownResultCodecError, match="Register it in _result_config"):
        load_result_bundle(obj._result_dir, ResultConfig.default())


def test_type_rule_can_select_codec_without_type_registration() -> None:
    obj = TypeRuleProbe()

    assert obj.load_or_create() == TypeRuleValue("rule-based")
    assert _manifest(obj)["root"]["$furu"]["codec"] == "test.type-rule.v1"


def test_object_protocol_round_trips() -> None:
    obj = ProtocolProbe()

    assert obj.load_or_create() == ProtocolValue("protocol")
    root = _manifest(obj)["root"]["$furu"]
    assert root["codec"] == "furu.object-protocol.v1"
    assert root["python_type"] == "test_results.ProtocolValue"


def test_optional_numpy_codec() -> None:
    pytest.importorskip("numpy")
    np = import_module("numpy")
    obj = NumpyProbe()

    result = obj.load_or_create()
    assert np.array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    root = _manifest(obj)["root"]["$furu"]
    assert root["codec"] == "numpy.ndarray.npy.v1"
    assert (obj._result_dir / "artifacts" / "value.npy").exists()

    lazy_result = NumpyProbe(lazy_mode=True).load_or_create()
    assert isinstance(lazy_result, furu.LazyValue)
    assert np.array_equal(
        lazy_result.load(),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_numpy_object_dtype_requires_custom_codec() -> None:
    pytest.importorskip("numpy")
    with pytest.raises(ResultSerializationError, match="Object-dtype NumPy arrays"):
        NumpyObjectProbe().load_or_create()


def test_optional_polars_codec() -> None:
    pytest.importorskip("polars")
    pl = import_module("polars")
    obj = PolarsProbe()

    result = cast(Any, obj.load_or_create())
    assert result.equals(pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}))
    root = _manifest(obj)["root"]["$furu"]
    assert root["codec"] == "polars.DataFrame.parquet.v1"
    assert (obj._result_dir / "artifacts" / "value.parquet").exists()

    lazy_result = PolarsProbe(lazy_mode=True).load_or_create()
    assert isinstance(lazy_result, furu.LazyValue)
    assert cast(Any, lazy_result.load()).equals(
        pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    )


def test_path_derived_directories_mirror_logical_paths() -> None:
    obj = NestedLazyPathProbe()

    result = cast(dict[str, Any], obj.load_or_create())
    assert isinstance(result["model"]["layers"][0], furu.LazyValue)
    assert isinstance(result["model"]["layers"][1], furu.LazyValue)
    assert (
        obj._result_dir / "artifacts" / "model" / "layers" / "000000" / "value.json"
    ).exists()
    assert (
        obj._result_dir / "artifacts" / "model" / "layers" / "000001" / "value.json"
    ).exists()
    assert not list(obj._result_dir.glob("artifacts/model__layers__*"))


def test_unsupported_object_error_is_path_aware() -> None:
    with pytest.raises(
        ResultSerializationError,
        match=r'furu\.result.*ResultCodec.*FuruResult.*result\["bad"\]',
    ):
        UnsupportedProbe().load_or_create()


def test_cycles_raise_path_aware_error() -> None:
    with pytest.raises(ResultSerializationError, match="Cycles are not supported"):
        CycleProbe().load_or_create()


def test_failed_save_cleanup_leaves_no_completed_bundle() -> None:
    obj = FailingSaveProbe()

    with pytest.raises(RuntimeError, match="codec write failed"):
        obj.load_or_create()

    assert not obj._result_manifest_path.exists()
    assert not obj._result_dir.exists()
