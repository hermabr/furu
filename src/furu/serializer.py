from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Self, TypeGuard, final, get_origin

from furu._typing import annotated_metadata, strip_annotated
from furu.config import get_config
from furu.utils import JsonValue, fully_qualified_name, resolve_fully_qualified_name

_SERIALIZER_ATTR = "__furu_serializer__"


class ArtifactSerializer[T](ABC):
    @final
    @classmethod
    def _serializer_id(cls) -> str:
        return fully_qualified_name(cls)

    @classmethod
    def dependencies_available(cls) -> bool:
        return True

    @classmethod
    @abstractmethod
    def matches(cls, value: object) -> bool:
        pass

    @classmethod
    @abstractmethod
    def matches_type(cls, declared_type: object) -> bool:
        pass

    @classmethod
    @abstractmethod
    def schema(cls, declared_type: object) -> JsonValue:
        pass

    @classmethod
    @abstractmethod
    def dump(cls, value: T, *, declared_type: object) -> JsonValue:
        pass

    @classmethod
    @abstractmethod
    def load(cls, value: JsonValue, *, declared_type: object) -> T:
        pass


def _is_serializer_class(value: object) -> TypeGuard[type[ArtifactSerializer]]:
    return isinstance(value, type) and issubclass(value, ArtifactSerializer)


def _type_label(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def annotated_serializer(declared_type: object) -> type[ArtifactSerializer] | None:
    for metadata in annotated_metadata(declared_type):
        if _is_serializer_class(metadata):
            return metadata
    return None


def class_serializer(cls: type) -> type[ArtifactSerializer] | None:
    provider = getattr(cls, _SERIALIZER_ATTR, None)
    if provider is None:
        return None
    if _is_serializer_class(provider):
        return provider
    if callable(provider):
        serializer = provider()
        if _is_serializer_class(serializer):
            return serializer

    raise TypeError(
        f"{_type_label(cls)}.{_SERIALIZER_ATTR} must be an ArtifactSerializer "
        "class or a zero-argument callable returning one"
    )


def resolve_serializer(serializer_id: str) -> type[ArtifactSerializer]:
    serializer = resolve_fully_qualified_name(serializer_id)
    if not _is_serializer_class(serializer):
        raise TypeError(f"{serializer_id} is not an ArtifactSerializer")
    return serializer


@dataclass(frozen=True)
class ArtifactSerializerRegistry:
    serializers: tuple[type[ArtifactSerializer], ...] = ()

    def register(self, serializer: type[ArtifactSerializer]) -> Self:
        return type(self)(serializers=(serializer, *self.serializers))

    def find_serializer(self, value: object) -> type[ArtifactSerializer] | None:
        for serializer in self.serializers:
            if serializer.matches(value):
                return serializer
        return None

    def find_serializer_for_type(
        self,
        declared_type: object,
    ) -> type[ArtifactSerializer] | None:
        for serializer in self.serializers:
            if serializer.matches_type(declared_type):
                return serializer
        return None

    @classmethod
    @cache
    def default(cls) -> ArtifactSerializerRegistry:
        configured_serializers: list[type[ArtifactSerializer]] = []
        for serializer_id in get_config().serializers:
            serializer = resolve_serializer(serializer_id)
            if serializer.dependencies_available():
                configured_serializers.append(serializer)
        return cls(serializers=tuple(configured_serializers))


def serializer_for_type(
    declared_type: object,
    *,
    registry: ArtifactSerializerRegistry,
) -> type[ArtifactSerializer] | None:
    if serializer := annotated_serializer(declared_type):
        return serializer

    declared_type = strip_annotated(declared_type)
    if isinstance(declared_type, type):
        if serializer := class_serializer(declared_type):
            return serializer

    origin = get_origin(declared_type)
    if isinstance(origin, type):
        if serializer := class_serializer(origin):
            return serializer

    return registry.find_serializer_for_type(declared_type)


def serializer_for_value(
    value: object,
    *,
    declared_type: object,
    registry: ArtifactSerializerRegistry,
) -> type[ArtifactSerializer] | None:
    if serializer := annotated_serializer(declared_type):
        return serializer

    declared_type = strip_annotated(declared_type)
    if isinstance(declared_type, type):
        if serializer := class_serializer(declared_type):
            return serializer

    if serializer := class_serializer(type(value)):
        return serializer

    return registry.find_serializer(value)
