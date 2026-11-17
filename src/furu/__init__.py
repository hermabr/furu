from importlib.metadata import version

from furu._batched import batched
from furu.core import Missing, Spec
from furu.dependencies import dependency
from furu.execution.load_or_create import build, create, load_existing
from furu.logging import get_logger
from furu.migration.steps import (
    Added,
    MigrationStep,
    MovedFrom,
    Renamed,
)
from furu.provenance import Provenance, submitting_repo_root
from furu.resources import Worker
from furu.result.codec import Codec
from furu.result.ref import Ref, ref
from furu.serializer.registry import Serializer
from furu.spec_metadata import Metadata, Throttle
from furu.storage.directory import SpecDirectory
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
    "Serializer",
    "Spec",
    "SpecDirectory",
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
    "submitting_repo_root",
]
