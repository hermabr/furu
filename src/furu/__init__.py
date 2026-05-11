from importlib.metadata import version

from furu.core import Furu
from furu.dependencies import dependency
from furu.executor import InMemoryScheduler, execute_local, run_worker
from furu.execution import BlockedOnDependencies, load_or_create
from furu.logging import get_logger
from furu.migration import Migration
from furu.result import LazyResult, save_as
from furu.result.codec import ResultCodec, ResultRegistry
from furu.validate import validate

__version__ = version("furu")

__all__ = [
    "__version__",
    "BlockedOnDependencies",
    "Furu",
    "LazyResult",
    "Migration",
    "dependency",
    "ResultCodec",
    "ResultRegistry",
    "get_logger",
    "InMemoryScheduler",
    "load_or_create",
    "execute_local",
    "run_worker",
    "save_as",
    "validate",
]
