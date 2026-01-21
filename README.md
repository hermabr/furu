# furu

> **Note:** `v0.0.x` is alpha and may include breaking changes.

A Python library for building cacheable, nested pipelines. Define computations as config objects; furu turns configs into stable on-disk artifact directories, records metadata/state, and reuses results across runs.

Built on [chz](https://github.com/openai/chz) for declarative configs.

## Installation

```bash
uv add "furu[dashboard]"
```

Or with pip:

```bash
pip install "furu[dashboard]"
```

The `[dashboard]` extra includes the web dashboard. Omit it for the core library only.

## Quickstart

1. Subclass `furu.Furu[T]`
2. Implement `_create(self) -> T` (compute and write to `self.furu_dir`)
3. Implement `_load(self) -> T` (load from `self.furu_dir`)
4. Call `load_or_create()`

```python
# my_project/pipelines.py
import json
from pathlib import Path
import furu

class TrainModel(furu.Furu[Path]):
    lr: float = furu.chz.field(default=1e-3)
    steps: int = furu.chz.field(default=1000)

    def _create(self) -> Path:
        # Write outputs into the artifact directory
        (self.furu_dir / "metrics.json").write_text(
            json.dumps({"lr": self.lr, "steps": self.steps})
        )
        ckpt = self.furu_dir / "checkpoint.bin"
        ckpt.write_bytes(b"...")
        return ckpt

    def _load(self) -> Path:
        # Load outputs back from disk
        return self.furu_dir / "checkpoint.bin"
```

```python
# run_train.py
from my_project.pipelines import TrainModel

# First call: runs _create(), caches result
artifact = TrainModel(lr=3e-4, steps=5000).load_or_create()

# Second call with same config: loads from cache via _load()
artifact = TrainModel(lr=3e-4, steps=5000).load_or_create()
```

> **Tip:** Define Furu classes in importable modules (not `__main__`); the artifact namespace is derived from the class's module + qualified name.

## Core Concepts

### How Caching Works

Each `Furu` instance maps deterministically to a directory based on its config:

```
<root>/<namespace>/<hash>/
```

- **namespace**: Derived from the class's module + qualified name (e.g., `my_project.pipelines/TrainModel`)
- **hash**: Computed from the object's config values using Blake2s

When you call `load_or_create()`:
1. If no cached result exists → run `_create()`, save state as "success"
2. If cached result exists → run `_load()` to retrieve it
3. If another process is running → wait for it to finish, then load

### Nested Pipelines (Dependencies)

Furu objects compose via nested configs. Each dependency gets its own artifact folder:

```python
import furu

class Dataset(furu.Furu[str]):
    name: str = furu.chz.field(default="toy")

    def _create(self) -> str:
        (self.furu_dir / "data.txt").write_text("hello\nworld\n")
        return "ready"

    def _load(self) -> str:
        return (self.furu_dir / "data.txt").read_text()


class TrainTextModel(furu.Furu[str]):
    dataset: Dataset = furu.chz.field(default_factory=Dataset)

    def _create(self) -> str:
        data = self.dataset.load_or_create()  # Triggers Dataset cache
        (self.furu_dir / "model.txt").write_text(f"trained on:\n{data}")
        return "trained"

    def _load(self) -> str:
        return (self.furu_dir / "model.txt").read_text()
```

### Storage Structure

```
$FURU_PATH/
├── data/                         # Default storage (version_controlled=False)
│   └── <module>/<Class>/
│       └── <hash>/
│           ├── .furu/
│           │   ├── metadata.json # Config, git info, environment
│           │   ├── state.json    # Status and timestamps
│           │   ├── furu.log    # Captured logs
│           │   └── SUCCESS.json  # Marker file
│           └── <your outputs>    # Files from _create()
└── raw/                          # Shared directory for large files

<project-root>/
├── pyproject.toml
├── .gitignore                    # Includes furu-data/artifacts/
└── furu-data/
    └── artifacts/                # version_controlled=True (override via FURU_VERSION_CONTROLLED_PATH)
        └── <same structure>
```

## Features

### FuruList: Managing Experiment Collections

`FuruList` provides a collection interface for organizing related experiments:

```python
import furu

class MyExperiments(furu.FuruList[TrainModel]):
    baseline = TrainModel(lr=1e-3, steps=1000)
    fast_lr = TrainModel(lr=1e-2, steps=1000)
    long_run = TrainModel(lr=1e-3, steps=10000)

    # Can also use a dict for dynamic configs
    configs = {
        "tiny": TrainModel(lr=1e-3, steps=100),
        "huge": TrainModel(lr=1e-4, steps=100000),
    }

# Iterate over all experiments
for exp in MyExperiments:
    exp.load_or_create()

# Access by name
exp = MyExperiments.by_name("baseline")

# Get all as list
all_exps = MyExperiments.all()

# Get (name, instance) pairs
for name, exp in MyExperiments.items():
    print(f"{name}: {exp.exists()}")
```

### Custom Validation

Override `_validate()` to add custom cache invalidation logic:

```python
class ModelWithValidation(furu.Furu[Path]):
    checkpoint_name: str = "model.pt"

    def _validate(self) -> bool:
        # Return False to force re-computation
        ckpt = self.furu_dir / self.checkpoint_name
        return ckpt.exists() and ckpt.stat().st_size > 0

    def _create(self) -> Path:
        ...

    def _load(self) -> Path:
        ...
```

### Checking State Without Loading

```python
obj = TrainModel(lr=3e-4, steps=5000)

# Check if cached result exists (runs _validate())
if obj.exists():
    print("Already computed!")

# Get metadata without triggering computation
metadata = obj.get_metadata()
print(f"Hash: {obj._furu_hash}")
print(f"Dir: {obj.furu_dir}")
```

### Serialization

Furu objects can be serialized to/from dictionaries:

```python
obj = TrainModel(lr=3e-4, steps=5000)

# Serialize to dict (for storage, transmission)
data = obj.to_dict()

# Reconstruct from dict
obj2 = TrainModel.from_dict(data)

# Get Python code representation (useful for logging)
print(obj.to_python())
# Output: TrainModel(lr=0.0003, steps=5000)
```

### Raw Directory

For large files that shouldn't be versioned per-config, use the shared raw directory:

```python
class LargeDataProcessor(furu.Furu[Path]):
    def _create(self) -> Path:
        # self.raw_dir is shared across all configs
        # Create a subfolder for isolation if needed
        my_raw = self.raw_dir / self._furu_hash
        my_raw.mkdir(exist_ok=True)
        
        large_file = my_raw / "huge_dataset.bin"
        # ... write large file ...
        return large_file
```

### Version-Controlled Storage

For artifacts that should be stored separately (e.g., checked into git):

```python
class VersionedConfig(furu.Furu[dict], version_controlled=True):
    # Stored under <project>/furu-data/artifacts by default.
    # Override with FURU_VERSION_CONTROLLED_PATH if needed.
    ...
```

It is typical to keep `furu-data/data/` and `furu-data/raw/` in `.gitignore` while
committing `furu-data/artifacts/`.

## Logging

Furu installs stdlib `logging` handlers that capture logs to per-artifact files.

```python
import logging
import furu

log = logging.getLogger(__name__)

class MyPipeline(furu.Furu[str]):
    def _create(self) -> str:
        log.info("Starting computation...")  # Goes to furu.log
        log.debug("Debug details...")
        return "done"
```

### Console Output

By default, furu logs to console using Rich in a compact format:

```
HHMMSS file.py:line message
```

Furu emits status messages like:
```
load_or_create TrainModel abc123def (missing->create)
load_or_create TrainModel abc123def (success->load)
```

### Explicit Setup

```python
import furu

# Eagerly install logging handlers (optional, happens automatically)
furu.configure_logging()

# Get the furu logger
logger = furu.get_logger()
```

## Error Handling

```python
from furu import FuruComputeError, FuruWaitTimeout, FuruLockNotAcquired

try:
    result = obj.load_or_create()
except FuruComputeError as e:
    print(f"Computation failed: {e}")
    print(f"State file: {e.state_path}")
    print(f"Original error: {e.original_error}")
except FuruWaitTimeout:
    print("Timed out waiting for another process")
except FuruLockNotAcquired:
    print("Could not acquire lock")
```

## Submitit Integration

Run computations on SLURM clusters via [submitit](https://github.com/facebookincubator/submitit):

```python
import submitit
import furu

executor = submitit.AutoExecutor(folder="submitit_logs")
executor.update_parameters(
    timeout_min=60,
    slurm_partition="gpu",
    gpus_per_node=1,
)

# Submit job and return immediately
job = my_furu_obj.load_or_create(executor=executor)

# Job ID is tracked in .furu/state.json
print(job.job_id)
```

Furu handles preemption, requeuing, and state tracking automatically.

## Dashboard

The web dashboard provides experiment browsing, filtering, and dependency visualization.

### Running the Dashboard

```bash
# Full dashboard with React frontend
furu-dashboard serve

# Or with options
furu-dashboard serve --host 0.0.0.0 --port 8000 --reload

# API server only (no frontend)
furu-dashboard api
```

Or via Python:
```bash
python -m furu.dashboard serve
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/experiments` | List experiments with filtering/pagination |
| `GET /api/experiments/{namespace}/{hash}` | Get experiment details |
| `GET /api/experiments/{namespace}/{hash}/relationships` | Get dependencies |
| `GET /api/stats` | Aggregate statistics |
| `GET /api/dag` | Dependency graph for visualization |

### Filtering

The `/api/experiments` endpoint supports:

- `result_status`: `absent`, `incomplete`, `success`, `failed`
- `attempt_status`: `queued`, `running`, `success`, `failed`, `cancelled`, `preempted`, `crashed`
- `namespace`: Filter by namespace prefix
- `backend`: `local`, `submitit`
- `hostname`, `user`: Filter by execution environment
- `started_after`, `started_before`: ISO datetime filters
- `config_filter`: Filter by config field (e.g., `lr=0.001`)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FURU_PATH` | `./data-furu/` | Base storage directory for non-versioned artifacts |
| `FURU_VERSION_CONTROLLED_PATH` | `<project>/furu-data/artifacts` | Override version-controlled storage root |
| `FURU_LOG_LEVEL` | `INFO` | Console verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `FURU_IGNORE_DIFF` | `false` | Skip embedding git diff in metadata |
| `FURU_ALWAYS_RERUN` | `""` | Comma-separated class qualnames to always rerun (use `ALL` to bypass cache globally; cannot combine with other entries; entries must be importable) |
| `FURU_POLL_INTERVAL_SECS` | `10` | Polling interval for queued/running jobs |
| `FURU_WAIT_LOG_EVERY_SECS` | `10` | Interval between "waiting" log messages |
| `FURU_STALE_AFTER_SECS` | `1800` | Consider running jobs stale after this duration |
| `FURU_LEASE_SECS` | `120` | Compute lock lease duration |
| `FURU_HEARTBEAT_SECS` | `lease/3` | Heartbeat interval for running jobs |
| `FURU_PREEMPT_MAX` | `5` | Maximum submitit requeues on preemption |
| `FURU_CANCELLED_IS_PREEMPTED` | `false` | Treat SLURM CANCELLED as preempted |
| `FURU_RICH_UNCAUGHT_TRACEBACKS` | `true` | Use Rich for exception formatting |

Local `.env` files are loaded automatically if `python-dotenv` is installed.

### Programmatic Configuration

```python
import furu
from pathlib import Path

# Set/get root directory
furu.set_furu_root(Path("/my/storage"))
root = furu.get_furu_root()

# Access config directly
furu.FURU_CONFIG.ignore_git_diff = True
furu.FURU_CONFIG.poll_interval = 5.0
```

### Class-Level Options

```python
class MyPipeline(furu.Furu[Path], version_controlled=True):
    _max_wait_time_sec = 3600.0  # Wait up to 1 hour (default: 600)
    ...
```

## Metadata

Each artifact records:

| Category | Fields |
|----------|--------|
| **Config** | `furu_python_def`, `furu_obj`, `furu_hash`, `furu_path` |
| **Git** | `git_commit`, `git_branch`, `git_remote`, `git_patch`, `git_submodules` |
| **Environment** | `timestamp`, `command`, `python_version`, `executable`, `platform`, `hostname`, `user`, `pid` |

Access via:
```python
metadata = obj.get_metadata()
print(metadata.git_commit)
print(metadata.hostname)
```

## Public API

```python
from furu import (
    # Core
    Furu,
    FuruList,
    FURU_CONFIG,
    
    # Configuration
    get_furu_root,
    set_furu_root,
    
    # Errors
    FuruError,
    FuruComputeError,
    FuruLockNotAcquired,
    FuruWaitTimeout,
    MISSING,
    
    # Serialization
    FuruSerializer,
    
    # Storage
    StateManager,
    MetadataManager,
    
    # Runtime
    configure_logging,
    get_logger,
    load_env,
    
    # Adapters
    SubmititAdapter,
    
    # Re-exports
    chz,
    submitit,
    
    # Version
    __version__,
)
```

## Non-goals / Caveats

- **Prototype status**: APIs and on-disk formats may change
- **Not a workflow scheduler** (for now): It's a lightweight caching layer for Python code
- **No distributed coordination**: Lock files work on shared filesystems but aren't distributed
