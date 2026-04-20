from importlib.metadata import version

from furu.core import Furu
from furu.execution import load_or_create
from furu.logging import get_logger
from furu.results import (
    FuruLazy,
    FuruResult,
    ResultCodec,
    ResultConfig,
    ResultRegistry,
    SaveWith,
    at,
    default_result_config,
    default_result_registry,
    lazy,
    result,
    save_with,
)
from furu.validate import validate

__version__ = version("furu")

__all__ = [
    "__version__",
    "Furu",
    "FuruLazy",
    "FuruResult",
    "ResultCodec",
    "ResultConfig",
    "ResultRegistry",
    "SaveWith",
    "at",
    "default_result_config",
    "default_result_registry",
    "get_logger",
    "lazy",
    "load_or_create",
    "result",
    "save_with",
    "validate",
]
