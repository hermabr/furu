import importlib
from typing import Any, cast


def _get_pydantic_base_model() -> type[Any] | None:
    try:
        module = importlib.import_module("pydantic")
    except Exception:
        return None
    base_model = getattr(module, "BaseModel", None)
    if isinstance(base_model, type):
        return cast(type[Any], base_model)
    return None


BaseModel = _get_pydantic_base_model()
