from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Self, final, get_args, get_origin

from furu._declared_types import strip_annotated
from furu.config import get_config
from furu.utils import JsonValue, fully_qualified_name, resolve_fully_qualified_name


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


def _annotated_serializer(declared_type: object) -> type[ArtifactSerializer] | None:
    if get_origin(declared_type) is not Annotated:
        return None

    for metadata in get_args(declared_type)[1:]:
        if isinstance(metadata, type) and issubclass(metadata, ArtifactSerializer):
            return metadata
    return None


def _class_serializer(cls: type) -> type[ArtifactSerializer] | None:
    _SERIALIZER_ATTR = "__furu_serializer__"
    provider = getattr(cls, _SERIALIZER_ATTR, None)
    if provider is None:
        return None
    if isinstance(provider, type) and issubclass(provider, ArtifactSerializer):
        return provider
    if callable(provider):
        serializer = provider()
        if isinstance(serializer, type) and issubclass(serializer, ArtifactSerializer):
            return serializer

    raise TypeError(
        f"{cls.__module__}.{cls.__qualname__}.{_SERIALIZER_ATTR} must be an ArtifactSerializer "
        "class or a zero-argument callable returning one"
    )


@dataclass(frozen=True)
class ArtifactSerializerRegistry:
    serializers: tuple[type[ArtifactSerializer], ...] = ()

    def with_serializer(self, serializer: type[ArtifactSerializer]) -> Self:
        return type(self)(serializers=(serializer, *self.serializers))

    def serializer_for_schema(
        self, declared_type: object
    ) -> type[ArtifactSerializer] | None:
        if serializer := _annotated_serializer(declared_type):
            return serializer

        declared_type = strip_annotated(declared_type)
        if isinstance(declared_type, type):
            if serializer := _class_serializer(declared_type):
                return serializer

        origin = get_origin(declared_type)
        if isinstance(origin, type):
            if serializer := _class_serializer(origin):
                return serializer

        for serializer in self.serializers:
            if serializer.matches_type(declared_type):
                return serializer
        return None

    def serializer_for_dump(
        self,
        value: object,
        *,
        declared_type: object,
    ) -> type[ArtifactSerializer] | None:
        if serializer := _annotated_serializer(declared_type):
            return serializer

        declared_type = strip_annotated(declared_type)
        if isinstance(declared_type, type):
            if serializer := _class_serializer(declared_type):
                return serializer

        if serializer := _class_serializer(type(value)):
            return serializer

        for serializer in self.serializers:
            if serializer.matches(value):
                return serializer
        return None

    @classmethod
    @cache
    def default(cls) -> ArtifactSerializerRegistry:
        configured_serializers: list[type[ArtifactSerializer]] = []
        for serializer_id in get_config().serializers:
            serializer = resolve_fully_qualified_name(serializer_id)
            if not (
                isinstance(serializer, type)
                and issubclass(serializer, ArtifactSerializer)
            ):
                raise TypeError(f"{serializer_id} is not an ArtifactSerializer")
            if serializer.dependencies_available():
                configured_serializers.append(serializer)
        return cls(serializers=tuple(configured_serializers))
