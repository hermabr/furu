from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import pytest

import furu
from furu import Furu
from furu.constants import FIELDSMARKER
from furu.metadata import ArtifactSpec
from furu.utils import JsonValue


def _fields(node: JsonValue) -> dict[str, Any]:
    assert isinstance(node, dict)
    fields = node[FIELDSMARKER]
    assert isinstance(fields, dict)
    return fields


class _Run(Furu[int]):
    value: int
    gpus: Annotated[int, furu.skip] = 1

    def create(self) -> int:
        return self.value


class _BadSkip(Furu[int]):
    gpus: Annotated[int, furu.skip]

    def create(self) -> int:
        return self.gpus


@furu.function
def _skip_param_run(value: int, gpus: Annotated[int, furu.skip] = 1) -> int:
    return value


@furu.function
def _skip_param_no_default(value: int, gpus: Annotated[int, furu.skip]) -> int:
    return value


@dataclass(frozen=True)
class _Cfg:
    lr: float
    gpus: Annotated[int, furu.skip] = 1


class _NestedRun(Furu[float]):
    cfg: _Cfg

    def create(self) -> float:
        return self.cfg.lr


def test_skipped_field_excluded_from_artifact_and_schema() -> None:
    obj = _Run(value=3, gpus=8)

    assert "value" in _fields(obj._artifact_data)
    assert "gpus" not in _fields(obj._artifact_data)
    assert "gpus" not in _fields(obj._schema_data)

    # The skipped field is still an ordinary, usable attribute.
    assert obj.gpus == 8


def test_skipped_field_does_not_affect_hash_or_object_id() -> None:
    a = _Run(value=3, gpus=1)
    b = _Run(value=3, gpus=99)

    assert a._artifact_hash == b._artifact_hash
    assert a._artifact_schema_hash == b._artifact_schema_hash
    assert a.object_id == b.object_id

    # A non-skipped field still changes the artifact hash.
    assert _Run(value=4, gpus=1)._artifact_hash != a._artifact_hash


def test_skipped_field_roundtrips_to_default_when_loaded() -> None:
    obj = _Run(value=3, gpus=8)

    loaded = _Run.from_artifact(ArtifactSpec.from_furu(obj))

    assert loaded.value == 3
    # Skipped fields are absent from the artifact, so the default is restored.
    assert loaded.gpus == 1


def test_skipped_field_without_default_raises() -> None:
    with pytest.raises(TypeError, match="must declare a default"):
        _ = _BadSkip(gpus=1).object_id


def test_skip_on_function_parameter() -> None:
    obj = _skip_param_run.as_furu(value=3, gpus=8)

    assert "gpus" not in _fields(obj._artifact_data)
    assert "gpus" not in _fields(obj._schema_data)
    assert getattr(obj, "gpus") == 8


def test_skip_function_parameter_without_default_raises() -> None:
    with pytest.raises(TypeError, match="must declare a default"):
        _ = _skip_param_no_default.as_furu(value=3, gpus=1).object_id


def test_skip_on_nested_dataclass_field() -> None:
    a = _NestedRun(cfg=_Cfg(lr=0.1, gpus=1))
    b = _NestedRun(cfg=_Cfg(lr=0.1, gpus=8))

    assert a._artifact_hash == b._artifact_hash
    assert a._artifact_schema_hash == b._artifact_schema_hash

    cfg_artifact = _fields(a._artifact_data)["cfg"]
    assert "gpus" not in _fields(cfg_artifact)

    loaded = _NestedRun.from_artifact(ArtifactSpec.from_furu(a))
    assert loaded.cfg == _Cfg(lr=0.1, gpus=1)
