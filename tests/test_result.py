from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Annotated, Any, ClassVar, cast
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ConfigDict
from runs import run_of

import furu
from furu import Ref, Spec
from furu._declared_types import child_declared_type
from furu.result.bundle import (
    _save_result_bundle as _save_result_bundle_impl,
)
from furu.result.bundle import (
    load_result_bundle as load_result_bundle_impl,
)
from furu.result.codec import (
    Codec,
    CodecMeta,
    NumpyNpyCodec,
)
from furu.storage._layout import data_dir_in, result_dir_in

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")


def _save_rows(
    values: list[Any],
    bundle_dir: Path,
    *,
    declared_type: object = Any,
    result_codecs: tuple[type[Codec], ...] = (),
    label: str = "Rows",
) -> None:
    _save_result_bundle_impl(
        values,
        bundle_dir,
        declared_type=declared_type,
        result_codecs=result_codecs,
        data_dir=data_dir_in(bundle_dir.parent),
        label=label,
    )


def _load_rows(
    bundle_dir: Path, rows: list[int], *, declared_type: object = Any
) -> list[Any]:
    return load_result_bundle_impl(
        bundle_dir,
        rows,
        data_dir=data_dir_in(bundle_dir.parent),
        declared_type=declared_type,
    )


def _save_result_bundle(
    value: object,
    bundle_dir: Path,
    *,
    declared_type: object = Any,
    result_codecs: tuple[type[Codec], ...],
) -> None:
    _save_rows(
        [value], bundle_dir, declared_type=declared_type, result_codecs=result_codecs
    )


def load_result_bundle(bundle_dir: Path, *, declared_type: object = Any) -> object:
    return _load_rows(bundle_dir, [0], declared_type=declared_type)[0]


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


def test_json_only_result_is_one_manifest_and_is_not_recomputed() -> None:
    JsonResult.create_calls.clear()
    obj = JsonResult()
    expected = {"metrics": {"loss": 0.12, "ok": True}, "items": [1, 2, None, "x"]}

    assert not obj.done
    assert obj.create() == expected
    assert obj.done
    assert obj.create() == expected
    assert obj.load_existing() == expected
    assert len(JsonResult.create_calls) == 1
    result_dir = result_dir_in(run_of(obj))
    assert [path.name for path in result_dir.iterdir()] == ["manifest.json"]


class PathResult(Spec[dict[str, Path]]):
    def create(self) -> dict[str, Path]:
        return {
            "relative": Path("outputs/model.bin"),
            "absolute": Path("/tmp/furu/model.bin"),
        }


def test_path_values_round_trip() -> None:
    assert PathResult().create() == {
        "relative": Path("outputs/model.bin"),
        "absolute": Path("/tmp/furu/model.bin"),
    }


class DataPathResult(Spec[Path]):
    def create(self) -> Path:
        path = self.directory.data / "model.bin"
        path.write_text("weights", encoding="utf-8")
        return path


def test_data_dir_path_follows_publish() -> None:
    obj = DataPathResult()

    result = obj.create()

    version = run_of(obj)
    assert result == data_dir_in(version).resolve() / "model.bin"
    assert result.read_text(encoding="utf-8") == "weights"
    assert obj.load_existing() == result


def test_tuple_set_and_frozenset_round_trip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = {
        "tuple": (1, "x", Path("model.bin")),
        "set": {3, 1, 2},
        "frozenset": frozenset({"b", "a"}),
    }

    _save_result_bundle(value, bundle_dir, result_codecs=())

    assert _same(load_result_bundle(bundle_dir), value)


def test_set_of_values_without_value_based_repr_is_rejected(tmp_path: Path) -> None:
    class AddressRepr:
        __hash__ = object.__hash__

    bundle_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="no value-based repr"):
        _save_result_bundle(
            {AddressRepr(), AddressRepr()}, bundle_dir, result_codecs=()
        )


class NonFiniteFloatResult(Spec[dict[str, float]]):
    def create(self) -> dict[str, float]:
        return {
            "nan": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
        }


def test_non_finite_floats_round_trip() -> None:
    result = NonFiniteFloatResult().create()

    assert math.isnan(result["nan"])
    assert result["pos_inf"] == float("inf")
    assert result["neg_inf"] == float("-inf")


def test_datetime_values_round_trip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = {
        "naive": datetime(2026, 9, 1, 12, 30, 45, 123456),
        "aware": datetime(2026, 9, 1, 12, 30, tzinfo=UTC),
        "fixed": datetime(
            2026,
            9,
            1,
            12,
            30,
            tzinfo=timezone(timedelta(hours=5, minutes=30)),
        ),
    }

    _save_result_bundle(value, bundle_dir, result_codecs=())

    loaded = cast(dict[str, datetime], load_result_bundle(bundle_dir))
    assert loaded == value
    assert _files(bundle_dir) == {"manifest.json"}


class _CustomTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta:
        return timedelta(hours=1)

    def tzname(self, dt: datetime | None) -> str:
        return "custom"

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)


class _DatetimeSubclass(datetime):
    pass


class _CustomTimezoneDatetimeCodec(Codec[datetime]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, datetime)

    @classmethod
    def save(cls, value: datetime, artifact_directory: Path) -> Mapping[str, object]:
        return {"value": value.replace(tzinfo=None).isoformat()}

    @classmethod
    def load(cls, metadata: Mapping[str, object], artifact_directory: Path) -> datetime:
        value = datetime.fromisoformat(cast(str, metadata["value"]))
        return value.replace(tzinfo=_CustomTimezone())


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 9, 1, fold=1),
        datetime(
            2026,
            9,
            1,
            tzinfo=timezone(timedelta(hours=5, minutes=30), "IST"),
        ),
        datetime(2026, 9, 1, tzinfo=ZoneInfo("America/Los_Angeles")),
        datetime(2026, 9, 1, tzinfo=_CustomTimezone()),
        _DatetimeSubclass(2026, 9, 1),
    ],
    ids=["fold", "timezone-name", "zoneinfo", "custom-timezone", "subclass"],
)
def test_datetime_requires_codec_when_iso_round_trip_is_lossy(
    tmp_path: Path, value: datetime
) -> None:
    with pytest.raises(ValueError, match="use an explicit custom codec"):
        _save_result_bundle(
            value,
            tmp_path / "bundle",
            result_codecs=(),
        )


def test_explicit_codec_overrides_native_datetime_storage(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    value = datetime(2026, 9, 1, 12, 30, tzinfo=_CustomTimezone())

    _save_result_bundle(
        value,
        bundle_dir,
        result_codecs=(_CustomTimezoneDatetimeCodec,),
    )

    loaded = load_result_bundle(bundle_dir)
    assert loaded == value
    assert isinstance(cast(datetime, loaded).tzinfo, _CustomTimezone)


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

    @classmethod
    def save(
        cls, value: _CountingValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        cls.dump_calls += 1
        artifact_directory.joinpath("value.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _CountingValue:
        cls.load_calls += 1
        return _CountingValue(
            int(artifact_directory.joinpath("value.txt").read_text(encoding="utf-8"))
        )


class _OtherCountingCodec(Codec[_CountingValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _CountingValue)

    @classmethod
    def save(
        cls, value: _CountingValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("other.txt").write_text("x", encoding="utf-8")
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _CountingValue:
        artifact_directory.joinpath("other.txt").read_text(encoding="utf-8")
        return _CountingValue(0)


class _AutoRegisteredValue:
    def __init__(self, value: int) -> None:
        self.value = value


class _AutoRegisteredValueCodec(Codec[_AutoRegisteredValue]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredValue)

    @classmethod
    def save(
        cls, value: _AutoRegisteredValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("auto.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(
            int(artifact_directory.joinpath("auto.txt").read_text(encoding="utf-8"))
        )


class _CoreRegistryAutoValueCodec(Codec[_AutoRegisteredValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredValue)

    @classmethod
    def save(
        cls, value: _AutoRegisteredValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("registry.txt").write_text(
            str(value.value),
            encoding="utf-8",
        )
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
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

    @classmethod
    def save(cls, value: Any, artifact_directory: Path) -> Mapping[str, object]:
        np.save(
            artifact_directory / "auto.npy",
            value.view(np.ndarray),
            allow_pickle=False,
        )
        return {}

    @classmethod
    def load(cls, metadata: Mapping[str, object], artifact_directory: Path) -> Any:
        return np.load(artifact_directory / "auto.npy", allow_pickle=False)


class _OptOutRegisteredValue:
    pass


class _OptOutRegisteredValueCodec(Codec[_OptOutRegisteredValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _OptOutRegisteredValue)

    @classmethod
    def save(
        cls, value: _OptOutRegisteredValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("manual.txt").write_text("", encoding="utf-8")
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
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

    @classmethod
    def save(
        cls, value: _DataDirPathValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        return {"path": value.path}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _DataDirPathValue:
        return _DataDirPathValue(cast(Path, metadata["path"]))


class RegistryAutoRegisteredValueResult(Spec[_AutoRegisteredValue]):
    result_codecs = (_CoreRegistryAutoValueCodec,)

    def create(self) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(10)


class AnnotatedAutoRegisteredValueResult(
    Spec[Annotated[_AutoRegisteredValue, _CoreRegistryAutoValueCodec]]
):
    def create(self) -> _AutoRegisteredValue:
        return _AutoRegisteredValue(11)


class _ScratchBackedValue:
    def __init__(self, source: Path) -> None:
        self.source = source


class _ScratchBackedCodec(Codec[_ScratchBackedValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _ScratchBackedValue)

    @classmethod
    def save(
        cls, value: _ScratchBackedValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        artifact_directory.joinpath("data.txt").write_text(
            value.source.read_text(encoding="utf-8"), encoding="utf-8"
        )
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _ScratchBackedValue:
        return _ScratchBackedValue(artifact_directory / "data.txt")


class ScratchBackedResult(Spec[_ScratchBackedValue]):
    name: str
    result_codecs = (_ScratchBackedCodec,)

    def create(self) -> _ScratchBackedValue:
        source = self.directory.scratch / "download.csv"
        source.write_text(self.name, encoding="utf-8")
        return _ScratchBackedValue(source)


def test_result_path_into_scratch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scratch dir"):
        _save_result_bundle(
            tmp_path / "scratch" / "download.csv",
            tmp_path / "result",
            result_codecs=(),
        )


def test_scratch_files_survive_until_codec_save() -> None:
    obj = ScratchBackedResult(name="lazy")

    value = obj.create()

    assert not (run_of(obj) / "scratch").exists()
    assert value.source.is_relative_to(result_dir_in(run_of(obj)))
    assert value.source.read_text(encoding="utf-8") == "lazy"


def test_result_codec_meta_find_codec_uses_result_codecs() -> None:
    first = (_CountingCodec,)
    second = (_CountingCodec, _OtherCountingCodec)

    assert CodecMeta.find_codec(_CountingValue(1), ()) is None
    assert CodecMeta.find_codec(_CountingValue(1), first) is _CountingCodec
    with pytest.raises(TypeError, match="result codecs matched multiple codecs"):
        CodecMeta.find_codec(_CountingValue(1), second)


def test_user_defined_codec_is_auto_registered(tmp_path: Path) -> None:
    assert (
        CodecMeta.find_codec(_AutoRegisteredValue(1), ()) is _AutoRegisteredValueCodec
    )

    bundle_dir = tmp_path / "bundle"
    _save_result_bundle(
        _AutoRegisteredValue(3),
        bundle_dir,
        result_codecs=(),
    )

    artifact_dir = bundle_dir / "rows" / "0" / "root"
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

    assert (bundle_dir / "rows" / "0" / "root" / "manual.txt").exists()
    loaded = load_result_bundle(bundle_dir)
    assert isinstance(loaded, _OptOutRegisteredValue)


def test_auto_registered_codec_takes_priority_over_builtin_codec(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    value = np.arange(3, dtype=np.int64).view(_AutoRegisteredArray)

    _save_result_bundle(value, bundle_dir, result_codecs=())

    artifact_dir = bundle_dir / "rows" / "0" / "root"
    assert (artifact_dir / "auto.npy").exists()
    assert not (artifact_dir / "data.npy").exists()


def test_task_result_codecs_take_priority_over_auto_registered_codec() -> None:
    obj = RegistryAutoRegisteredValueResult()

    loaded = obj.create()

    assert isinstance(loaded, _AutoRegisteredValue)
    assert loaded.value == 10
    artifact_dir = result_dir_in(run_of(obj)) / "rows" / "0" / "root"
    assert (artifact_dir / "registry.txt").exists()
    assert not (artifact_dir / "auto.txt").exists()


def test_annotated_codec_takes_priority_over_auto_registered_codec() -> None:
    obj = AnnotatedAutoRegisteredValueResult()

    loaded = obj.create()

    assert isinstance(loaded, _AutoRegisteredValue)
    assert loaded.value == 11
    artifact_dir = result_dir_in(run_of(obj)) / "rows" / "0" / "root"
    assert (artifact_dir / "registry.txt").exists()
    assert not (artifact_dir / "auto.txt").exists()


def test_codec_defined_after_default_codec_layers_cache_is_auto_registered() -> None:
    class LateAutoRegisteredValue:
        pass

    assert CodecMeta.find_codec(LateAutoRegisteredValue(), ()) is None

    class LateAutoRegisteredCodec(Codec[LateAutoRegisteredValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, LateAutoRegisteredValue)

        @classmethod
        def save(
            cls, value: LateAutoRegisteredValue, artifact_directory: Path
        ) -> Mapping[str, object]:
            artifact_directory.joinpath("late.txt").write_text("", encoding="utf-8")
            return {}

        @classmethod
        def load(
            cls, metadata: Mapping[str, object], artifact_directory: Path
        ) -> LateAutoRegisteredValue:
            artifact_directory.joinpath("late.txt").read_text(encoding="utf-8")
            return LateAutoRegisteredValue()

    assert (
        CodecMeta.find_codec(LateAutoRegisteredValue(), ()) is LateAutoRegisteredCodec
    )


def test_auto_registered_codecs_must_not_be_ambiguous() -> None:
    class AutoAmbiguousValue:
        pass

    class FirstAutoAmbiguousCodec(Codec[AutoAmbiguousValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, AutoAmbiguousValue)

        @classmethod
        def save(
            cls, value: AutoAmbiguousValue, artifact_directory: Path
        ) -> Mapping[str, object]:
            artifact_directory.joinpath("first.txt").write_text("", encoding="utf-8")
            return {}

        @classmethod
        def load(
            cls, metadata: Mapping[str, object], artifact_directory: Path
        ) -> AutoAmbiguousValue:
            artifact_directory.joinpath("first.txt").read_text(encoding="utf-8")
            return AutoAmbiguousValue()

    class SecondAutoAmbiguousCodec(Codec[AutoAmbiguousValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, AutoAmbiguousValue)

        @classmethod
        def save(
            cls, value: AutoAmbiguousValue, artifact_directory: Path
        ) -> Mapping[str, object]:
            artifact_directory.joinpath("second.txt").write_text("", encoding="utf-8")
            return {}

        @classmethod
        def load(
            cls, metadata: Mapping[str, object], artifact_directory: Path
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
    assert (result_dir_in(run_of(obj)) / "artifacts" / "weights" / "data.npy").exists()


def test_generic_furu_base_with_annotated_result_codec_is_rejected() -> None:
    obj = GenericAnnotatedArrayResult()

    with pytest.raises(TypeError, match="concrete result type directly as Spec"):
        obj.create()


def test_strict_pydantic_annotated_codec_selects_artifact_storage() -> None:
    obj = StrictAnnotatedArrayResult()
    loaded = obj.create()

    assert isinstance(loaded, StrictAnnotatedArrayOutput)
    assert np.array_equal(loaded.weights, np.arange(3, dtype=np.int64))
    assert (result_dir_in(run_of(obj)) / "artifacts" / "weights" / "data.npy").exists()

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
    result_codecs = (_DataDirPathCodec,)

    def create(self) -> dict[str, _DataDirPathValue]:
        path = self.directory.data / "data.zarr"
        path.mkdir()
        value = _DataDirPathValue(path)
        return {
            "first": value,
            "second": cast(_DataDirPathValue, furu.ref(value, codec=_DataDirPathCodec)),
        }


def test_ref_field_resolves_codec_from_registry_and_loads_on_demand() -> None:
    obj = RefArrayResult()
    created = obj.create()

    assert isinstance(created, RefArrayOutput)
    assert isinstance(created.weights, Ref)
    assert np.array_equal(created.weights.load(), np.arange(4))
    assert (result_dir_in(run_of(obj)) / "artifacts" / "weights" / "data.npy").exists()

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
    # Handed back from storage, but a Ref reads its artifact only on load().
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
    data_path = data_dir_in(run_of(obj)) / "data.zarr"

    assert loaded["first"].path.resolve() == data_path.resolve()
    # Data-dir paths move when the attempt is published, so the creating run
    # gets the stored value back, exactly as a later load would.
    assert isinstance(loaded["second"], _DataDirPathValue)
    assert loaded["second"].path.resolve() == data_path.resolve()

    assert not (
        result_dir_in(run_of(obj)) / "rows" / "0" / "first" / "data.zarr"
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


def test_unsupported_nested_path_includes_padded_index(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    layers = [{} for _ in range(10)]
    layers[3] = {"weights": _CustomTensor()}

    with pytest.raises(ValueError) as exc_info:
        _save_result_bundle({"layers": layers}, bundle_dir, result_codecs=())
    assert "layers/03/weights" in str(exc_info.value)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("$furu", "reserved"),
        (1, "must be strings"),
        ("bad/key", "artifact path segment"),
        ("", "artifact path segment"),
        ("..", "artifact path segment"),
    ],
)
def test_dict_keys_that_cannot_be_stored_fail(
    tmp_path: Path, key: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _save_result_bundle({key: "x"}, tmp_path / "bundle", result_codecs=())


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


@dataclass(frozen=True)
class NestedOuter:
    inner: NestedInner
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


def test_numpy_array_round_trips_as_a_plain_array() -> None:
    obj = NumpyResult()
    created = obj.create()

    assert (result_dir_in(run_of(obj)) / "artifacts" / "weights" / "data.npy").exists()
    for loaded in (created, obj.load_existing()):
        assert _same(loaded, {"weights": np.arange(10, dtype=np.float32)})


class PolarsResult(Spec[dict[str, object]]):
    def create(self) -> dict[str, object]:
        return {"frame": pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})}


def test_polars_dataframe_round_trips() -> None:
    obj = PolarsResult()
    loaded = obj.create()

    assert (
        result_dir_in(run_of(obj)) / "artifacts" / "frame" / "data.parquet"
    ).exists()
    assert isinstance(loaded, dict)
    frame = loaded["frame"]
    assert isinstance(frame, pl.DataFrame)
    assert frame.equals(pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}))


class NestedNumpyResult(Spec[dict[str, list[dict[str, object]]]]):
    def create(self) -> dict[str, list[dict[str, object]]]:
        return {
            "layers": [{"weights": np.arange(i, dtype=np.float32)} for i in range(10)]
        }


def test_nested_numpy_paths_use_list_length_padded_indexes() -> None:
    obj = NestedNumpyResult()
    loaded = obj.create()

    layers_dir = result_dir_in(run_of(obj)) / "artifacts" / "layers"
    for i in range(10):
        weights_file = layers_dir / f"{i:02d}" / "weights" / "data.npy"
        assert weights_file.exists(), f"missing {weights_file}"

    assert isinstance(loaded, dict)
    layers = loaded["layers"]
    assert len(layers) == 10
    for i, layer in enumerate(layers):
        assert isinstance(layer, dict)
        assert np.array_equal(layer["weights"], np.arange(i, dtype=np.float32))


def test_private_save_result_bundle_refuses_existing_directory(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    with pytest.raises(FileExistsError):
        _save_result_bundle({"x": 1}, bundle_dir, result_codecs=())


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

    loaded = load_result_bundle(bundle_dir, declared_type=Ref[_CountingValue])

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


def _files(bundle_dir: Path) -> set[str]:
    return {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file()
    }


def _same(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    match a:
        case np.ndarray():
            return a.dtype == b.dtype and np.array_equal(a, b)
        case pl.DataFrame():
            return a.schema == b.schema and a.equals(b)
        case dict():
            return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
        case list() | tuple():
            return len(a) == len(b) and all(map(_same, a, b))
        case LooResult():
            return _same(vars(a), vars(b))
        case _:
            return a == b


@dataclass(frozen=True)
class LooResult:
    accuracy: float
    preds: np.ndarray
    mistakes: pl.DataFrame
    extras: dict[str, tuple[np.ndarray, int]]


def _loo(seed: int) -> LooResult:
    return LooResult(
        accuracy=seed / 10,
        preds=np.full(4, seed, dtype=np.float32),
        # A table per row, of its own length: none for seed 0.
        mistakes=pl.DataFrame({"example": range(seed)}, schema={"example": pl.Int64}),
        extras={"hidden": (np.full((2, 2), seed), seed)},
    )


def test_plain_rows_share_one_manifest_and_load_in_the_order_asked(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    values = [
        {"seed": seed, "loss": seed / 10, "tags": ("a", seed)} for seed in range(5)
    ]

    _save_rows(values, bundle_dir)

    assert _files(bundle_dir) == {"manifest.json"}
    assert _load_rows(bundle_dir, [0, 1, 2, 3, 4]) == values
    assert _load_rows(bundle_dir, [3, 0, 3]) == [values[3], values[0], values[3]]
    assert _load_rows(bundle_dir, []) == []


def test_arrays_and_tables_of_every_row_share_one_file_per_position(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    values = [_loo(seed) for seed in range(6)]

    _save_rows(values, bundle_dir, declared_type=LooResult)

    assert _files(bundle_dir) == {
        "manifest.json",
        "artifacts/preds/data.npy",
        "artifacts/mistakes/data.parquet",
        "artifacts/extras/hidden/0/data.npy",
    }
    assert np.load(bundle_dir / "artifacts/preds/data.npy").shape == (6, 4)
    assert pl.read_parquet(bundle_dir / "artifacts/mistakes/data.parquet").height == 15
    assert _same(
        _load_rows(bundle_dir, list(range(6)), declared_type=LooResult), values
    )
    assert _same(
        _load_rows(bundle_dir, [5, 0, 2], declared_type=LooResult),
        [values[5], values[0], values[2]],
    )


def test_one_row_bundle_is_the_first_row_of_a_larger_one(tmp_path: Path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"

    _save_rows([_loo(3)], one)
    _save_rows([_loo(3), _loo(3)], two)

    assert (
        json.loads((one / "manifest.json").read_text())
        == (json.loads((two / "manifest.json").read_text())[:1])
    )
    assert _files(one) == _files(two)
    assert _same(_load_rows(one, [0]), [_loo(3)])


def test_rows_without_a_value_are_left_out_of_the_shared_file(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    values = [
        {"preds": None},
        {"preds": np.arange(3)},
        {"preds": None},
        {"preds": np.arange(3) + 1},
    ]

    _save_rows(values, bundle_dir)

    assert np.load(bundle_dir / "artifacts/preds/data.npy").shape == (2, 3)
    assert _same(_load_rows(bundle_dir, [3, 2, 1, 0]), values[::-1])


def test_values_that_do_not_merge_get_a_directory_per_row(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    values = [
        {
            "ragged": np.zeros(3),
            "schemas": pl.DataFrame({"x": [1]}),
            "codecs": np.ones(2),
            "shapes": np.ones(2),
            "merged": np.arange(2),
        },
        {
            "ragged": np.zeros(4),
            "schemas": pl.DataFrame({"x": [None]}),
            "codecs": pl.DataFrame({"x": [2]}),
            "shapes": {"inner": np.ones(2)},
            "merged": np.arange(2),
        },
    ]

    _save_rows(values, bundle_dir)

    assert _files(bundle_dir) == {
        "manifest.json",
        "artifacts/merged/data.npy",
        "rows/0/ragged/data.npy",
        "rows/1/ragged/data.npy",
        "rows/0/schemas/data.parquet",
        "rows/1/schemas/data.parquet",
        "rows/0/codecs/data.npy",
        "rows/1/codecs/data.parquet",
        "rows/0/shapes/data.npy",
        "rows/1/shapes/inner/data.npy",
    }
    assert _same(_load_rows(bundle_dir, [1, 0]), values[::-1])


def test_whole_result_value_only_clashes_with_a_field_named_root(
    tmp_path: Path,
) -> None:
    apart, clash = tmp_path / "apart", tmp_path / "clash"

    values = [np.zeros(3), {"x": np.zeros(3)}, {"x": np.ones(3)}]
    _save_rows(values, apart)
    _save_rows([np.zeros(3), {"root": np.ones(3)}], clash)

    assert _files(apart) == {
        "manifest.json",
        "artifacts/root/data.npy",
        "artifacts/x/data.npy",
    }
    assert _files(clash) == {
        "manifest.json",
        "rows/0/root/data.npy",
        "rows/1/root/data.npy",
    }
    assert _same(_load_rows(apart, [0, 1, 2]), values)
    assert _same(_load_rows(clash, [1, 0]), [{"root": np.ones(3)}, np.zeros(3)])


def test_zero_dimensional_arrays_keep_their_dtype_when_stacked(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    record = np.dtype([("a", "i4"), ("b", "f8")])
    values = [
        {
            "big_endian": np.array(seed, dtype=">i2"),
            "text": np.array("ab", dtype="U5"),
            "record": np.array((seed, 0.5), dtype=record),
        }
        for seed in range(2)
    ]

    _save_rows(values, bundle_dir)
    loaded = _load_rows(bundle_dir, [0, 1])

    assert _same(loaded, values)
    assert loaded[0]["record"].flags.writeable


@pytest.mark.filterwarnings("ignore:Stored array in format 3.0")
def test_arrays_that_one_file_cannot_hold_are_left_to_save(tmp_path: Path) -> None:
    named_dir = tmp_path / "named"
    named = [np.array([(seed,)], dtype=[("日本", "i4")]) for seed in range(2)]

    _save_rows(named, named_dir)

    assert _files(named_dir) == {
        "manifest.json",
        "rows/0/root/data.npy",
        "rows/1/root/data.npy",
    }
    assert _same(_load_rows(named_dir, [0, 1]), named)

    masked = np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    with pytest.raises(NotImplementedError):
        _save_rows([masked], tmp_path / "masked")


def test_codec_with_only_save_and_load_gets_one_call_per_row(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _CountingCodec.dump_calls = 0
    _CountingCodec.load_calls = 0

    _save_rows(
        [_CountingValue(seed) for seed in range(3)],
        bundle_dir,
        result_codecs=(_CountingCodec,),
    )
    loaded = _load_rows(bundle_dir, [2, 0])

    assert _files(bundle_dir) == {
        "manifest.json",
        "rows/0/root/value.txt",
        "rows/1/root/value.txt",
        "rows/2/root/value.txt",
    }
    assert not (bundle_dir / "artifacts").exists()
    assert [value.value for value in loaded] == [2, 0]
    assert (_CountingCodec.dump_calls, _CountingCodec.load_calls) == (3, 2)


class _MergingCodec(_CountingCodec):
    auto_register: ClassVar[bool] = False
    loaded_lines: ClassVar[list[list[object]]] = []

    @classmethod
    def save_many(
        cls, values: Sequence[_CountingValue], artifact_directory: Path
    ) -> Sequence[Mapping[str, object]] | None:
        artifact_directory.joinpath("values.txt").write_text(
            "\n".join(str(value.value) for value in values), encoding="utf-8"
        )
        return [{"line": line} for line in range(len(values))]

    @classmethod
    def load_many(
        cls, metadatas: Sequence[Mapping[str, object]], artifact_directory: Path
    ) -> Sequence[_CountingValue]:
        lines = artifact_directory.joinpath("values.txt").read_text().splitlines()
        cls.loaded_lines.append([metadata["line"] for metadata in metadatas])
        return [
            _CountingValue(int(lines[cast(int, metadata["line"])]))
            for metadata in metadatas
        ]


def test_codec_with_save_many_and_load_many_gets_one_call_each(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _MergingCodec.loaded_lines.clear()

    _save_rows(
        [_CountingValue(seed) for seed in range(4)],
        bundle_dir,
        result_codecs=(_MergingCodec,),
    )
    loaded = _load_rows(bundle_dir, [3, 1])

    assert _files(bundle_dir) == {"manifest.json", "artifacts/root/values.txt"}
    assert [value.value for value in loaded] == [3, 1]
    assert _MergingCodec.loaded_lines == [[3, 1]]

    # A Ref reads its own row, and only when asked.
    (lazy,) = _load_rows(bundle_dir, [2], declared_type=Ref[_CountingValue])
    assert _MergingCodec.loaded_lines == [[3, 1]]
    assert lazy.load().value == 2
    assert _MergingCodec.loaded_lines == [[3, 1], [2]]


class _Sweep(Spec[_CountingValue]):
    seed: int
    key: int = 0
    result_codecs = (_MergingCodec,)

    @furu.batched(lambda spec, worker: (spec.key, 100))
    def create(
        specs: list[_Sweep], directory: furu.SpecDirectory
    ) -> list[_CountingValue]:
        return [_CountingValue(spec.seed) for spec in specs]


class _Alone(Spec[_CountingValue]):
    seed: int
    result_codecs = (_MergingCodec,)

    def create(self) -> _CountingValue:
        return _CountingValue(self.seed)


def test_load_existing_reads_each_run_once_and_keeps_the_shape_it_was_given() -> None:
    first = [_Sweep(seed=seed) for seed in range(3)]
    second = [_Sweep(seed=seed, key=1) for seed in range(3, 5)]
    alone = _Alone(seed=9)
    furu.build([first, second, alone])
    _MergingCodec.loaded_lines.clear()

    loaded = furu.load_existing(
        {"a": (second[1], first[2]), "b": [alone, first[0], second[0]]}
    )

    assert {
        name: type(values)(value.value for value in values)
        for name, values in loaded.items()
    } == {"a": (4, 2), "b": [9, 0, 3]}
    # One call per run, with the rows asked for in the order asked.
    assert _MergingCodec.loaded_lines == [[1, 0], [2, 0], [0]]


def test_large_fallback_warns_once_per_label_and_field(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def ragged(seed: int) -> Any:
        return np.zeros(seed % 2)

    # Three per-row folders a row: 99 for 33 rows, 102 for 34.
    rows = [
        {"a": ragged(seed), "b": [ragged(seed), ragged(seed)]} for seed in range(34)
    ]

    with caplog.at_level(logging.WARNING, logger="furu"):
        _save_rows(rows[:33], tmp_path / "small", label="Ragged")
        assert caplog.messages == []

        _save_rows(rows, tmp_path / "first", label="Ragged")
        _save_rows(rows, tmp_path / "second", label="Ragged")
    assert caplog.messages == [
        "Ragged: 34 values in a were not merged into one file and got a "
        + "directory each",
        "Ragged: 68 values in b were not merged into one file and got a "
        + "directory each",
    ]
