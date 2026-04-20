from __future__ import annotations

from typing import Any

from furu.utils import class_label

from .paths import LogicalPath


def _type_label(runtime_type: type[Any] | None) -> str | None:
    if runtime_type is None:
        return None
    try:
        return class_label(runtime_type)
    except Exception:
        return runtime_type.__name__


class ResultSerializationError(TypeError):
    def __init__(
        self,
        logical_path: LogicalPath,
        runtime_type: type[Any] | None = None,
        detail: str | None = None,
    ) -> None:
        message = f"Cannot serialize value at {logical_path.display()}"
        type_label = _type_label(runtime_type)
        if type_label is not None:
            message += f" of type {type_label}"
        message += "."
        if detail is not None:
            message += f" {detail}"
        message += (
            " Register a result codec, wrap the value with furu.result(...), "
            "annotate the field with SaveWith(...), or implement "
            "__furu_result_dump__ / __furu_result_load__."
        )
        super().__init__(message)


class ResultLoadError(RuntimeError):
    def __init__(self, logical_path: LogicalPath, detail: str) -> None:
        super().__init__(f"Cannot load value at {logical_path.display()}. {detail}")


class ResultCodecError(RuntimeError):
    pass
