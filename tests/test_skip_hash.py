from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import furu
from furu.constants import FIELDSMARKER
from furu.dependencies import collect_declared_refs
from furu.metadata import ArtifactSpec
from furu.serializer.artifact import _from_json, to_json
from furu.serializer.schema import schema_type


def _fields(node: Any) -> dict[str, Any]:
    assert isinstance(node, dict)
    raw = node[FIELDSMARKER]
    assert isinstance(raw, dict)
    return raw


class _SkipScalar(furu.Furu[str]):
    important: str
    debug: Annotated[str, furu.skip_hash]

    def create(self) -> str:
        return self.important


@dataclass(frozen=True)
class _Config:
    important: int
    debug: Annotated[str, furu.skip_hash]


class _NestedSkip(furu.Furu[int]):
    config: _Config

    def create(self) -> int:
        return self.config.important


class _Dep(furu.Furu[str]):
    value: str

    def create(self) -> str:
        return self.value


class _SkipDep(furu.Furu[str]):
    important: str
    debug_dep: Annotated[_Dep, furu.skip_hash]

    def create(self) -> str:
        return self.important


@furu.function
def _scaled(value: int, run_id: Annotated[str, furu.skip_hash]) -> int:
    return value * 2


def test_skip_hash_field_is_kept_in_schema_and_artifact_data() -> None:
    obj = _SkipScalar(important="keep", debug="changes")

    assert _fields(obj._schema_data)["debug"] == "builtins.str"
    assert _fields(obj._artifact_data)["debug"] == "changes"


def test_skip_hash_field_excluded_from_hashes() -> None:
    a = _SkipScalar(important="keep", debug="one")
    b = _SkipScalar(important="keep", debug="two")

    assert a._artifact_hash == b._artifact_hash
    assert a._artifact_schema_hash == b._artifact_schema_hash
    assert a.object_id == b.object_id


def test_non_skip_field_still_changes_hash() -> None:
    a = _SkipScalar(important="keep", debug="one")
    c = _SkipScalar(important="different", debug="one")

    assert a._artifact_hash != c._artifact_hash


def test_skip_hash_field_omitted_from_hash_schema_only() -> None:
    full = schema_type(_SkipScalar, set(), artifact_serializers=())
    hashed = schema_type(_SkipScalar, set(), artifact_serializers=(), for_hash=True)

    assert "debug" in _fields(full)
    assert "debug" not in _fields(hashed)
    assert "important" in _fields(hashed)


def test_skip_hash_field_omitted_from_hash_artifact_only() -> None:
    obj = _SkipScalar(important="keep", debug="x")

    full = to_json(obj, declared_type=_SkipScalar, artifact_serializers=())
    hashed = to_json(
        obj, declared_type=_SkipScalar, artifact_serializers=(), for_hash=True
    )

    assert "debug" in _fields(full)
    assert "debug" not in _fields(hashed)
    assert "important" in _fields(hashed)


def test_skip_hash_on_nested_dataclass_field() -> None:
    a = _NestedSkip(config=_Config(important=1, debug="x"))
    b = _NestedSkip(config=_Config(important=1, debug="y"))

    assert a._artifact_hash == b._artifact_hash
    assert a._artifact_schema_hash == b._artifact_schema_hash

    config_fields = _fields(_fields(a._artifact_data)["config"])
    assert config_fields["debug"] == "x"


def test_skip_hash_round_trips_through_artifact() -> None:
    obj = _SkipScalar(important="keep", debug="restored")

    loaded = _from_json(obj._artifact_data)
    assert loaded == obj
    assert loaded.debug == "restored"

    artifact = ArtifactSpec.from_furu(obj)
    assert _SkipScalar.from_artifact(artifact) == obj


def test_skip_hash_dependency_excluded_from_hash_but_kept_as_dependency() -> None:
    a = _SkipDep(important="keep", debug_dep=_Dep(value="one"))
    b = _SkipDep(important="keep", debug_dep=_Dep(value="two"))

    assert a.object_id == b.object_id

    assert _fields(_fields(a._artifact_data)["debug_dep"])["value"] == "one"
    assert collect_declared_refs(a) == (_Dep(value="one"),)


def test_skip_hash_on_furu_function_parameter() -> None:
    a = _scaled.spec(2, "run-a")
    b = _scaled.spec(2, "run-b")

    assert a.object_id == b.object_id
    assert _fields(a._artifact_data)["run_id"] == "run-a"
    assert _fields(a._schema_data)["run_id"] == "builtins.str"
