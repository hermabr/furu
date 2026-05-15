from importlib.metadata import version

from furu.core import Furu
from furu.dependencies import dependency
from furu.execution import load_or_create
from furu.logging import get_logger
from furu.migration import Migration
from furu.resources import ResourceRequirements
from furu.result import LazyResult, save_as
from furu.result.codec import ResultCodec, ResultRegistry
from furu.validate import validate

__version__ = version("furu")

__all__ = [
    "__version__",
    "Furu",
    "LazyResult",
    "Migration",
    "ResourceRequirements",
    "dependency",
    "ResultCodec",
    "ResultRegistry",
    "get_logger",
    "load_or_create",
    "save_as",
    "validate",
]
