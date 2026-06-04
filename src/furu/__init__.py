from importlib.metadata import version

from furu.core import Furu
from furu.dependencies import dependency
from furu.logging import get_logger
from furu.function import function
from furu.migration import Migration
from furu.resources import ResourceRequirements
from furu.result import LazyResult, save_as
from furu.result.codec import ResultCodec, ResultRegistry
from furu.serializer import ArtifactSerializer, SerializerRegistry
from furu.validate import validate

__version__ = version("furu")

__all__ = [
    "__version__",
    "Furu",
    "ArtifactSerializer",
    "LazyResult",
    "Migration",
    "ResourceRequirements",
    "dependency",
    "function",
    "ResultCodec",
    "ResultRegistry",
    "SerializerRegistry",
    "get_logger",
    "save_as",
    "validate",
]
