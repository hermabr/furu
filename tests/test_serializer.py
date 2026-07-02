from __future__ import annotations

from typing import Annotated, Any, ClassVar

import pytest

from furu import Serializer, Spec
from furu.constants import (
    FIELDSMARKER,
    KINDMARKER,
    SCHEMAMARKER,
    SERIALIZERMARKER,
    VALUEMARKER,
)
from furu.metadata import ArtifactSpec
from furu.serializer.artifact import _from_json
from furu.serializer.registry import SerializerMeta
from furu.utils import JsonValue


class _Secret:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Secret) and self.value == other.value


class _ClassHookSecret(_Secret):
    @classmethod
    def __furu_serializer__(cls) -> type[Serializer[_Secret]]:
        return _HexSecretSerializer


class _HexSecretSerializer(Serializer[_Secret]):
    auto_register: ClassVar[bool] = False

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


class _DecimalSecretSerializer(Serializer[_Secret]):
    auto_register: ClassVar[bool] = False

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
    auto_register: ClassVar[bool] = False

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


class _AnnotatedSecretRun(Spec[int]):
    secret: Annotated[_Secret, _HexSecretSerializer]

    def create(self) -> int:
        return self.secret.value


class _RegistrySecretRun(Spec[int]):
    secret: _Secret

    @property
    def artifact_serializers(self) -> tuple[type[Serializer], ...]:
        return (_RegistrySecretSerializer,)

    def create(self) -> int:
        return self.secret.value


class _ClassHookSecretRun(Spec[int]):
    secret: _ClassHookSecret

    def create(self) -> int:
        return self.secret.value


class _TopLevelSerializedRun(Spec[int]):
    value: int

    @classmethod
    def __furu_serializer__(cls) -> type[Serializer[_TopLevelSerializedRun]]:
        return _TopLevelRunSerializer

    def create(self) -> int:
        return self.value


class _TopLevelRunSerializer(Serializer[_TopLevelSerializedRun]):
    auto_register: ClassVar[bool] = False

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


class _AutoRegisteredValue:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _AutoRegisteredValue) and self.value == other.value


class _AutoRegisteredValueSerializer(Serializer[_AutoRegisteredValue]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredValue)

    @classmethod
    def matches_type(cls, declared_type: object) -> bool:
        return declared_type is _AutoRegisteredValue

    @classmethod
    def schema(cls, declared_type: object) -> JsonValue:
        return {"type": "auto-value", "format": "auto"}

    @classmethod
    def dump(
        cls,
        value: _AutoRegisteredValue,
        *,
        declared_type: object,
    ) -> JsonValue:
        return {"auto": value.value}

    @classmethod
    def load(
        cls,
        value: JsonValue,
        *,
        declared_type: object,
    ) -> _AutoRegisteredValue:
        if not isinstance(value, dict) or not isinstance(value.get("auto"), int):
            raise ValueError("expected auto value artifact")
        return _AutoRegisteredValue(value["auto"])


class _RegistryAutoRegisteredValueSerializer(Serializer[_AutoRegisteredValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _AutoRegisteredValue)

    @classmethod
    def matches_type(cls, declared_type: object) -> bool:
        return declared_type is _AutoRegisteredValue

    @classmethod
    def schema(cls, declared_type: object) -> JsonValue:
        return {"type": "auto-value", "format": "registry"}

    @classmethod
    def dump(
        cls,
        value: _AutoRegisteredValue,
        *,
        declared_type: object,
    ) -> JsonValue:
        return {"registry": value.value}

    @classmethod
    def load(
        cls,
        value: JsonValue,
        *,
        declared_type: object,
    ) -> _AutoRegisteredValue:
        if not isinstance(value, dict) or not isinstance(value.get("registry"), int):
            raise ValueError("expected registry auto value artifact")
        return _AutoRegisteredValue(value["registry"])


class _OptOutRegisteredValue:
    pass


class _OptOutRegisteredValueSerializer(Serializer[_OptOutRegisteredValue]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _OptOutRegisteredValue)

    @classmethod
    def matches_type(cls, declared_type: object) -> bool:
        return declared_type is _OptOutRegisteredValue

    @classmethod
    def schema(cls, declared_type: object) -> JsonValue:
        return {"type": "opt-out-value"}

    @classmethod
    def dump(
        cls,
        value: _OptOutRegisteredValue,
        *,
        declared_type: object,
    ) -> JsonValue:
        return {"opt-out": True}

    @classmethod
    def load(
        cls,
        value: JsonValue,
        *,
        declared_type: object,
    ) -> _OptOutRegisteredValue:
        if not isinstance(value, dict) or value.get("opt-out") is not True:
            raise ValueError("expected opt-out value artifact")
        return _OptOutRegisteredValue()


class _AutoRegisteredValueRun(Spec[int]):
    value: _AutoRegisteredValue

    def create(self) -> int:
        return self.value.value


class _RegistryAutoRegisteredValueRun(Spec[int]):
    value: _AutoRegisteredValue

    @property
    def artifact_serializers(self) -> tuple[type[Serializer], ...]:
        return (_RegistryAutoRegisteredValueSerializer,)

    def create(self) -> int:
        return self.value.value


class _AnnotatedAutoRegisteredValueRun(Spec[int]):
    value: Annotated[
        _AutoRegisteredValue,
        _RegistryAutoRegisteredValueSerializer,
    ]

    def create(self) -> int:
        return self.value.value


def _custom_schema(
    serializer: type[Serializer],
    schema: JsonValue,
) -> JsonValue:
    return {
        KINDMARKER: "custom",
        SERIALIZERMARKER: serializer._serializer_id(),
        SCHEMAMARKER: schema,
    }


def _custom_artifact(
    serializer: type[Serializer],
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


def test_furu_artifact_serializers_define_schema_and_artifact() -> None:
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
    assert Spec.from_artifact(artifact) == obj


def test_user_defined_serializer_is_auto_registered() -> None:
    assert (
        SerializerMeta.serializer_for_schema(_AutoRegisteredValue, ())
        is _AutoRegisteredValueSerializer
    )
    assert (
        SerializerMeta.serializer_for_dump(
            _AutoRegisteredValue(1),
            declared_type=_AutoRegisteredValue,
            artifact_serializers=(),
        )
        is _AutoRegisteredValueSerializer
    )

    obj = _AutoRegisteredValueRun(value=_AutoRegisteredValue(42))

    assert _field(obj._schema_data, "value") == _custom_schema(
        _AutoRegisteredValueSerializer,
        {"type": "auto-value", "format": "auto"},
    )
    assert _field(obj._artifact_data, "value") == _custom_artifact(
        _AutoRegisteredValueSerializer,
        {"auto": 42},
    )
    assert _from_json(obj._artifact_data) == obj


def test_auto_register_false_opts_out_of_auto_registered_serializers() -> None:
    assert (
        SerializerMeta.serializer_for_schema(_OptOutRegisteredValue, ()) is None
    )
    assert (
        SerializerMeta.serializer_for_dump(
            _OptOutRegisteredValue(),
            declared_type=_OptOutRegisteredValue,
            artifact_serializers=(),
        )
        is None
    )

    assert (
        SerializerMeta.serializer_for_schema(
            _OptOutRegisteredValue,
            (_OptOutRegisteredValueSerializer,),
        )
        is _OptOutRegisteredValueSerializer
    )
    assert (
        SerializerMeta.serializer_for_dump(
            _OptOutRegisteredValue(),
            declared_type=_OptOutRegisteredValue,
            artifact_serializers=(_OptOutRegisteredValueSerializer,),
        )
        is _OptOutRegisteredValueSerializer
    )


def test_artifact_serializers_must_not_be_ambiguous() -> None:
    artifact_serializers = (_HexSecretSerializer, _DecimalSecretSerializer)

    with pytest.raises(
        TypeError,
        match="artifact serializers matched multiple serializers for schema",
    ) as schema_error:
        SerializerMeta.serializer_for_schema(_Secret, artifact_serializers)

    schema_message = str(schema_error.value)
    assert "_HexSecretSerializer" in schema_message
    assert "_DecimalSecretSerializer" in schema_message

    with pytest.raises(
        TypeError,
        match="artifact serializers matched multiple serializers for dump",
    ) as dump_error:
        SerializerMeta.serializer_for_dump(
            _Secret(1),
            declared_type=_Secret,
            artifact_serializers=artifact_serializers,
        )

    dump_message = str(dump_error.value)
    assert "_HexSecretSerializer" in dump_message
    assert "_DecimalSecretSerializer" in dump_message


def test_furu_artifact_serializers_take_priority_over_auto_registered_serializer() -> (
    None
):
    obj = _RegistryAutoRegisteredValueRun(value=_AutoRegisteredValue(42))

    assert _field(obj._schema_data, "value") == _custom_schema(
        _RegistryAutoRegisteredValueSerializer,
        {"type": "auto-value", "format": "registry"},
    )
    assert _field(obj._artifact_data, "value") == _custom_artifact(
        _RegistryAutoRegisteredValueSerializer,
        {"registry": 42},
    )
    assert _from_json(obj._artifact_data) == obj


def test_annotated_serializer_takes_priority_over_auto_registered_serializer() -> None:
    obj = _AnnotatedAutoRegisteredValueRun(value=_AutoRegisteredValue(42))

    assert _field(obj._schema_data, "value") == _custom_schema(
        _RegistryAutoRegisteredValueSerializer,
        {"type": "auto-value", "format": "registry"},
    )
    assert _field(obj._artifact_data, "value") == _custom_artifact(
        _RegistryAutoRegisteredValueSerializer,
        {"registry": 42},
    )
    assert _from_json(obj._artifact_data) == obj


def test_serializer_defined_after_default_cache_is_auto_registered() -> None:
    class LateAutoRegisteredValue:
        pass

    assert (
        SerializerMeta.serializer_for_schema(LateAutoRegisteredValue, ())
        is None
    )
    assert (
        SerializerMeta.serializer_for_dump(
            LateAutoRegisteredValue(),
            declared_type=LateAutoRegisteredValue,
            artifact_serializers=(),
        )
        is None
    )

    class LateAutoRegisteredValueSerializer(
        Serializer[LateAutoRegisteredValue]
    ):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, LateAutoRegisteredValue)

        @classmethod
        def matches_type(cls, declared_type: object) -> bool:
            return declared_type is LateAutoRegisteredValue

        @classmethod
        def schema(cls, declared_type: object) -> JsonValue:
            return {"type": "late-auto-value"}

        @classmethod
        def dump(
            cls,
            value: LateAutoRegisteredValue,
            *,
            declared_type: object,
        ) -> JsonValue:
            return {"late": True}

        @classmethod
        def load(
            cls,
            value: JsonValue,
            *,
            declared_type: object,
        ) -> LateAutoRegisteredValue:
            if not isinstance(value, dict) or value.get("late") is not True:
                raise ValueError("expected late auto value artifact")
            return LateAutoRegisteredValue()

    assert (
        SerializerMeta.serializer_for_schema(LateAutoRegisteredValue, ())
        is LateAutoRegisteredValueSerializer
    )
    assert (
        SerializerMeta.serializer_for_dump(
            LateAutoRegisteredValue(),
            declared_type=LateAutoRegisteredValue,
            artifact_serializers=(),
        )
        is LateAutoRegisteredValueSerializer
    )


def test_auto_registered_serializers_must_not_be_ambiguous() -> None:
    class AutoAmbiguousValue:
        pass

    class FirstAutoAmbiguousSerializer(Serializer[AutoAmbiguousValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, AutoAmbiguousValue)

        @classmethod
        def matches_type(cls, declared_type: object) -> bool:
            return declared_type is AutoAmbiguousValue

        @classmethod
        def schema(cls, declared_type: object) -> JsonValue:
            return {"type": "auto-ambiguous-value", "serializer": "first"}

        @classmethod
        def dump(
            cls,
            value: AutoAmbiguousValue,
            *,
            declared_type: object,
        ) -> JsonValue:
            return {"serializer": "first"}

        @classmethod
        def load(
            cls,
            value: JsonValue,
            *,
            declared_type: object,
        ) -> AutoAmbiguousValue:
            return AutoAmbiguousValue()

    class SecondAutoAmbiguousSerializer(Serializer[AutoAmbiguousValue]):
        @classmethod
        def matches(cls, value: object) -> bool:
            return isinstance(value, AutoAmbiguousValue)

        @classmethod
        def matches_type(cls, declared_type: object) -> bool:
            return declared_type is AutoAmbiguousValue

        @classmethod
        def schema(cls, declared_type: object) -> JsonValue:
            return {"type": "auto-ambiguous-value", "serializer": "second"}

        @classmethod
        def dump(
            cls,
            value: AutoAmbiguousValue,
            *,
            declared_type: object,
        ) -> JsonValue:
            return {"serializer": "second"}

        @classmethod
        def load(
            cls,
            value: JsonValue,
            *,
            declared_type: object,
        ) -> AutoAmbiguousValue:
            return AutoAmbiguousValue()

    with pytest.raises(
        TypeError,
        match="auto-registered serializer registry matched multiple serializers for schema",
    ) as schema_error:
        SerializerMeta.serializer_for_schema(AutoAmbiguousValue, ())

    schema_message = str(schema_error.value)
    assert "FirstAutoAmbiguousSerializer" in schema_message
    assert "SecondAutoAmbiguousSerializer" in schema_message

    with pytest.raises(
        TypeError,
        match="auto-registered serializer registry matched multiple serializers for dump",
    ) as dump_error:
        SerializerMeta.serializer_for_dump(
            AutoAmbiguousValue(),
            declared_type=AutoAmbiguousValue,
            artifact_serializers=(),
        )

    dump_message = str(dump_error.value)
    assert "FirstAutoAmbiguousSerializer" in dump_message
    assert "SecondAutoAmbiguousSerializer" in dump_message
