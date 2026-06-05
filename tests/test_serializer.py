from __future__ import annotations

from typing import Annotated, Any

from furu import ArtifactSerializer, ArtifactSerializerRegistry, Furu
from furu.constants import (
    FIELDSMARKER,
    KINDMARKER,
    SCHEMAMARKER,
    SERIALIZERMARKER,
    VALUEMARKER,
)
from furu.metadata import ArtifactSpec
from furu.serializer.artifact import _from_json
from furu.utils import JsonValue


class _Secret:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Secret) and self.value == other.value


class _ClassHookSecret(_Secret):
    @classmethod
    def __furu_serializer__(cls) -> type[ArtifactSerializer[_Secret]]:
        return _HexSecretSerializer


class _HexSecretSerializer(ArtifactSerializer[_Secret]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return type(value) is _Secret

    @classmethod
    def matches_type(cls, declared_type: object) -> bool:
        return declared_type is _Secret

    @classmethod
    def schema(cls, declared_type: object) -> JsonValue:
        return {"type": "secret", "format": "hex"}

    @classmethod
    def dump(cls, value: _Secret, *, declared_type: object) -> JsonValue:
        return {"hex": hex(value.value)}

    @classmethod
    def load(cls, value: JsonValue, *, declared_type: object) -> _Secret:
        if not isinstance(value, dict) or not isinstance(value.get("hex"), str):
            raise ValueError("expected hex secret artifact")

        loaded_type = declared_type if isinstance(declared_type, type) else _Secret
        return loaded_type(int(value["hex"], 16))


class _DecimalSecretSerializer(ArtifactSerializer[_Secret]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return type(value) is _Secret

    @classmethod
    def matches_type(cls, declared_type: object) -> bool:
        return declared_type is _Secret

    @classmethod
    def schema(cls, declared_type: object) -> JsonValue:
        return {"type": "secret", "format": "decimal"}

    @classmethod
    def dump(cls, value: _Secret, *, declared_type: object) -> JsonValue:
        return {"decimal": value.value}

    @classmethod
    def load(cls, value: JsonValue, *, declared_type: object) -> _Secret:
        if not isinstance(value, dict) or not isinstance(value.get("decimal"), int):
            raise ValueError("expected decimal secret artifact")
        return _Secret(value["decimal"])


class _RegistrySecretSerializer(_DecimalSecretSerializer):
    @classmethod
    def schema(cls, declared_type: object) -> JsonValue:
        return {"type": "secret", "format": "registry"}

    @classmethod
    def dump(cls, value: _Secret, *, declared_type: object) -> JsonValue:
        return {"registry": value.value}

    @classmethod
    def load(cls, value: JsonValue, *, declared_type: object) -> _Secret:
        if not isinstance(value, dict) or not isinstance(value.get("registry"), int):
            raise ValueError("expected registry secret artifact")
        return _Secret(value["registry"])


class _AnnotatedSecretRun(Furu[int]):
    secret: Annotated[_Secret, _HexSecretSerializer]

    def create(self) -> int:
        return self.secret.value


class _RegistrySecretRun(Furu[int]):
    secret: _Secret

    @property
    def serializer_registry(self) -> ArtifactSerializerRegistry:
        return super().serializer_registry.with_serializer(_RegistrySecretSerializer)

    def create(self) -> int:
        return self.secret.value


class _ClassHookSecretRun(Furu[int]):
    secret: _ClassHookSecret

    def create(self) -> int:
        return self.secret.value


class _TopLevelSerializedRun(Furu[int]):
    value: int

    @classmethod
    def __furu_serializer__(cls) -> type[ArtifactSerializer[_TopLevelSerializedRun]]:
        return _TopLevelRunSerializer

    def create(self) -> int:
        return self.value


class _TopLevelRunSerializer(ArtifactSerializer[_TopLevelSerializedRun]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return type(value) is _TopLevelSerializedRun

    @classmethod
    def matches_type(cls, declared_type: object) -> bool:
        return declared_type is _TopLevelSerializedRun

    @classmethod
    def schema(cls, declared_type: object) -> JsonValue:
        return {"type": "top-level-run", "version": 1}

    @classmethod
    def dump(
        cls,
        value: _TopLevelSerializedRun,
        *,
        declared_type: object,
    ) -> JsonValue:
        return {"doubled": value.value * 2}

    @classmethod
    def load(
        cls,
        value: JsonValue,
        *,
        declared_type: object,
    ) -> _TopLevelSerializedRun:
        if not isinstance(value, dict) or not isinstance(value.get("doubled"), int):
            raise ValueError("expected top-level run artifact")
        loaded_type = (
            declared_type
            if isinstance(declared_type, type)
            and issubclass(declared_type, _TopLevelSerializedRun)
            else _TopLevelSerializedRun
        )
        return loaded_type(value=value["doubled"] // 2)


def _custom_schema(
    serializer: type[ArtifactSerializer],
    schema: JsonValue,
) -> JsonValue:
    return {
        KINDMARKER: "custom",
        SERIALIZERMARKER: serializer._serializer_id(),
        SCHEMAMARKER: schema,
    }


def _custom_artifact(
    serializer: type[ArtifactSerializer],
    value: JsonValue,
) -> JsonValue:
    return {
        KINDMARKER: "custom",
        SERIALIZERMARKER: serializer._serializer_id(),
        VALUEMARKER: value,
    }


def _field(node: JsonValue, name: str) -> Any:
    assert isinstance(node, dict)
    fields = node[FIELDSMARKER]
    assert isinstance(fields, dict)
    return fields[name]


def test_annotated_serializer_defines_schema_and_artifact() -> None:
    obj = _AnnotatedSecretRun(secret=_Secret(42))

    assert _field(obj._schema_data, "secret") == _custom_schema(
        _HexSecretSerializer,
        {"type": "secret", "format": "hex"},
    )
    assert _field(obj._artifact_data, "secret") == _custom_artifact(
        _HexSecretSerializer,
        {"hex": "0x2a"},
    )
    assert _from_json(obj._artifact_data) == obj


def test_furu_artifact_serializer_registry_defines_schema_and_artifact() -> None:
    obj = _RegistrySecretRun(secret=_Secret(42))

    assert _field(obj._schema_data, "secret") == _custom_schema(
        _RegistrySecretSerializer,
        {"type": "secret", "format": "registry"},
    )
    assert _field(obj._artifact_data, "secret") == _custom_artifact(
        _RegistrySecretSerializer,
        {"registry": 42},
    )
    assert _RegistrySecretRun.from_artifact(ArtifactSpec.from_furu(obj)) == obj


def test_class_serializer_hook_defines_schema_and_artifact() -> None:
    obj = _ClassHookSecretRun(secret=_ClassHookSecret(42))

    assert _field(obj._schema_data, "secret") == _custom_schema(
        _HexSecretSerializer,
        {"type": "secret", "format": "hex"},
    )
    assert _field(obj._artifact_data, "secret") == _custom_artifact(
        _HexSecretSerializer,
        {"hex": "0x2a"},
    )
    loaded = _from_json(obj._artifact_data)
    assert loaded == obj
    assert isinstance(loaded.secret, _ClassHookSecret)


def test_furu_class_serializer_hook_can_replace_top_level_artifact() -> None:
    obj = _TopLevelSerializedRun(value=21)

    assert obj._schema_data == _custom_schema(
        _TopLevelRunSerializer,
        {"type": "top-level-run", "version": 1},
    )
    assert obj._artifact_data == _custom_artifact(
        _TopLevelRunSerializer,
        {"doubled": 42},
    )
    artifact = ArtifactSpec.from_furu(obj)
    assert _TopLevelSerializedRun.from_artifact(artifact) == obj
    assert Furu.from_artifact(artifact) == obj
