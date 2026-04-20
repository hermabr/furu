from importlib.metadata import version

from furu.core import Furu
from furu.execution import load_or_create
from furu.logging import get_logger
from furu.results import LazyValue, ResultRegistry
from furu.results.api import (
    FuruResult,
    ResultConfig,
    SaveWith,
    lazy,
    result,
    result_at,
    result_when_type,
)
from furu.results.codecs import ResultCodec
from furu.validate import validate

__version__ = version("furu")

__all__ = [
    "__version__",
    "Furu",
    "FuruResult",
    "LazyValue",
    "ResultCodec",
    "ResultConfig",
    "ResultRegistry",
    "SaveWith",
    "get_logger",
    "lazy",
    "load_or_create",
    "result",
    "result_at",
    "result_when_type",
    "validate",
]
