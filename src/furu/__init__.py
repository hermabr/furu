from importlib.metadata import version

from furu._declared_types import skip_hash
from furu._function import spec
from furu.core import Missing, Spec
from furu.dependencies import dependency
from furu.execution.create import create, load_existing
from furu.logging import get_logger
from furu.migration import Migration
from furu.resources import ResourceRequirements
from furu.result.lazy import LazyResult
from furu.result.save_as import save_as
from furu.result.codec import ResultCodec
from furu.serializer.registry import Serializer
from furu.utils import _install_main_module_alias
from furu.validate import validate

_install_main_module_alias()

__version__ = version("furu")

__all__ = [
    "__version__",
    "Spec",
    "Serializer",
    "LazyResult",
    "Migration",
    "Missing",
    "ResourceRequirements",
    "create",
    "dependency",
    "spec",
    "ResultCodec",
    "get_logger",
    "load_existing",
    "save_as",
    "skip_hash",
    "validate",
]
