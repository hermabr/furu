from importlib.metadata import version

from furu._batched import batched
from furu._declared_types import skip_hash
from furu.core import Missing, Spec
from furu.dependencies import dependency
from furu.execution.load_or_create import build, create, load_existing
from furu.logging import get_logger
from furu.migration.steps import (
    Added,
    MigrationStep,
    MovedFrom,
    Renamed,
    Retyped,
    Rewrite,
    Stale,
)
from furu.provenance import Provenance
from furu.resources import Worker
from furu.result.codec import Codec
from furu.result.ref import Ref, ref
from furu.serializer.registry import Serializer
from furu.spec_metadata import Metadata, Throttle
from furu.utils import _install_main_module_alias

_install_main_module_alias()

__version__ = version("furu")

__all__ = [
    "Added",
    "Codec",
    "Metadata",
    "MigrationStep",
    "Missing",
    "MovedFrom",
    "Provenance",
    "Ref",
    "Renamed",
    "Retyped",
    "Rewrite",
    "Serializer",
    "Spec",
    "Stale",
    "Throttle",
    "Worker",
    "__version__",
    "batched",
    "build",
    "create",
    "dependency",
    "get_logger",
    "load_existing",
    "ref",
    "skip_hash",
]
