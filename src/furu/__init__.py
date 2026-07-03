from importlib.metadata import version

from furu._declared_types import skip_hash
from furu._function import spec
from furu.core import Missing, Spec
from furu.dependencies import dependency
from furu.execution.load_or_create import create, load_existing
from furu.logging import get_logger
from furu.migration import Migration
from furu.resources import ResourceRequirements
from furu.result.codec import Codec
from furu.result.ref import Ref, ref
from furu.serializer.registry import Serializer
from furu.utils import _install_main_module_alias
from furu.validate import validate

_install_main_module_alias()

__version__ = version("furu")

__all__ = [
    "__version__",
    "Spec",
    "Serializer",
    "Codec",
    "Ref",
    "Migration",
    "Missing",
    "ResourceRequirements",
    "create",
    "dependency",
    "ref",
    "spec",
    "get_logger",
    "load_existing",
    "skip_hash",
    "validate",
]
