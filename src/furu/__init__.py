"""
Furu: cacheable, nested pipelines as config objects.

This package uses a src-layout. Import the package as `furu`.
"""

from importlib.metadata import version

import chz
import submitit

__version__ = version("furu")

from .config import FURU_CONFIG, FuruConfig, get_furu_root, set_furu_root
from .adapters import SubmititAdapter
from .core import Furu, FuruList
from .errors import (
    FuruComputeError,
    FuruError,
    FuruLockNotAcquired,
    FuruMigrationRequired,
    FuruWaitTimeout,
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
from .migrate import migrate
from .migration import (
    NamespacePair,
    MigrationCandidate,
    MigrationSkip,
    apply_migration,
    find_migration_candidates,
    find_migration_candidates_initialized_target,
)
from .serialization import FuruSerializer
from .storage import MetadataManager, StateManager

__all__ = [
    "__version__",
    "FURU_CONFIG",
    "Furu",
    "FuruComputeError",
    "FuruConfig",
    "FuruError",
    "FuruList",
    "FuruLockNotAcquired",
    "FuruMigrationRequired",
    "FuruSerializer",
    "FuruWaitTimeout",
    "MISSING",
    "migrate",
    "NamespacePair",
    "MigrationCandidate",
    "MigrationSkip",
    "apply_migration",
    "find_migration_candidates",
    "find_migration_candidates_initialized_target",
    "MetadataManager",
    "StateManager",
    "SubmititAdapter",
    "chz",
    "configure_logging",
    "current_holder",
    "current_log_dir",
    "enter_holder",
    "get_furu_root",
    "get_logger",
    "load_env",
    "log",
    "write_separator",
    "set_furu_root",
    "submitit",
]
