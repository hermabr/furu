from importlib.metadata import version

from furu._declared_types import skip_hash
from furu.core import Furu
from furu.dependencies import dependency
from furu.logging import get_logger
from furu.function import function
from furu.migration import Migration
from furu.resources import ResourceRequirements
from furu.result import LazyResult, save_as
from furu.result.codec import DataDirResultCodec, ResultCodec
from furu.serializer.registry import ArtifactSerializer
from furu.utils import _install_main_module_alias
from furu.validate import validate

_install_main_module_alias()

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
    "DataDirResultCodec",
    "ResultCodec",
    "get_logger",
    "save_as",
    "skip_hash",
    "validate",
]
