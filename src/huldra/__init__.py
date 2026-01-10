"""
Huldra: cacheable, nested pipelines as config objects.

This package uses a src-layout. Import the package as `huldra`.
"""

from importlib.metadata import version

import chz
import submitit

__version__ = version("huldra")

from .config import HULDRA_CONFIG, HuldraConfig, get_huldra_root, set_huldra_root
from .adapters import SubmititAdapter
from .core import Huldra, HuldraList
from .errors import (
    HuldraComputeError,
    HuldraError,
    HuldraLockNotAcquired,
    HuldraWaitTimeout,
    MISSING,
)
from .runtime import (
    configure_logging,
    current_holder,
    current_log_dir,
    enter_holder,
    get_logger,
    load_env,
    log,
    write_separator,
)
from .serialization import HuldraSerializer
from .storage import MetadataManager, StateManager

__all__ = [
    "__version__",
    "HULDRA_CONFIG",
    "Huldra",
    "HuldraComputeError",
    "HuldraConfig",
    "HuldraError",
    "HuldraList",
    "HuldraLockNotAcquired",
    "HuldraSerializer",
    "HuldraWaitTimeout",
    "MISSING",
    "MetadataManager",
    "StateManager",
    "SubmititAdapter",
    "chz",
    "configure_logging",
    "current_holder",
    "current_log_dir",
    "enter_holder",
    "get_huldra_root",
    "get_logger",
    "load_env",
    "log",
    "write_separator",
    "set_huldra_root",
    "submitit",
]
