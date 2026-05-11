from importlib.metadata import version

from furu.core import Furu
from furu.dependencies import dependency
from furu.execution import BlockedOnDependencies, load_or_create
from furu.local_executor import run_local_executor
from furu.logging import get_logger
from furu.migration import Migration
from furu.result import LazyResult, save_as
from furu.result.codec import ResultCodec, ResultRegistry
from furu.scheduler import Scheduler
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
    "Scheduler",
    "get_logger",
    "load_or_create",
    "run_local_executor",
    "save_as",
    "validate",
]
