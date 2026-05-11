from importlib.metadata import version

from furu.core import Furu
from furu.dependencies import dependency
from furu.execution import load_or_create
from furu.executor import (
    BlockedOnDependencies,
    ExecutionGraph,
    ExcessiveSuspensions,
    InMemoryScheduler,
    LocalExecutor,
    LocalExecutorFailed,
    Planner,
    run_local,
)
from furu.logging import get_logger
from furu.migration import Migration
from furu.result import LazyResult, save_as
from furu.result.codec import ResultCodec, ResultRegistry
from furu.validate import validate

__version__ = version("furu")

__all__ = [
    "__version__",
    "Furu",
    "LazyResult",
    "Migration",
    "BlockedOnDependencies",
    "ExecutionGraph",
    "ExcessiveSuspensions",
    "InMemoryScheduler",
    "LocalExecutor",
    "LocalExecutorFailed",
    "Planner",
    "dependency",
    "ResultCodec",
    "ResultRegistry",
    "get_logger",
    "load_or_create",
    "run_local",
    "save_as",
    "validate",
]
