from importlib.metadata import version

from furu.core import Furu
from furu.dependencies import dependency
from furu.logging import get_logger
from furu.method import furu_method
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
    "furu_method",
    "ResultCodec",
    "ResultRegistry",
    "get_logger",
    "save_as",
    "validate",
]
