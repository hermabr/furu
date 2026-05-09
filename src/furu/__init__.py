from importlib.metadata import PackageNotFoundError, version

from furu.core import CorruptResultError, Furu, MissingResultError
from furu.execution import load_or_create
from furu.logging import get_logger
from furu.result import LazyResult
from furu.validate import validate

try:
    __version__ = version("furu")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Furu",
    "CorruptResultError",
    "LazyResult",
    "MissingResultError",
    "get_logger",
    "load_or_create",
    "validate",
]
