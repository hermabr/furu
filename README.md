# gren

> **Note:** This is an early prototype; expect breaking changes.

A Python library for building cacheable, nested pipelines. Define computations as config objects; gren turns configs into stable on-disk artifact directories, records metadata/state, and reuses results across runs.

Built on [chz](https://github.com/openai/chz) for declarative configs.

## Installation

```bash
uv add "gren[dashboard]"
```

Or with pip:

```bash
pip install "gren[dashboard]"
```

The `[dashboard]` extra includes the web dashboard. Omit it for the core library only.

## Quickstart

1. Subclass `gren.Gren[T]`
2. Implement `_create(self) -> T` (compute and write to `self.gren_dir`)
3. Implement `_load(self) -> T` (load from `self.gren_dir`)
4. Call `load_or_create()`

```python
# my_project/pipelines.py
import json
from pathlib import Path
import gren

class TrainModel(gren.Gren[Path]):
    lr: float = gren.chz.field(default=1e-3)
    steps: int = gren.chz.field(default=1000)

    def _create(self) -> Path:
        # Write outputs into the artifact directory
        (self.gren_dir / "metrics.json").write_text(
            json.dumps({"lr": self.lr, "steps": self.steps})
        )
        ckpt = self.gren_dir / "checkpoint.bin"
        ckpt.write_bytes(b"...")
        return ckpt

    def _load(self) -> Path:
        # Load outputs back from disk
        return self.gren_dir / "checkpoint.bin"
```

```python
# run_train.py
from my_project.pipelines import TrainModel

# First call: runs _create(), caches result
artifact = TrainModel(lr=3e-4, steps=5000).load_or_create()

# Second call with same config: loads from cache via _load()
artifact = TrainModel(lr=3e-4, steps=5000).load_or_create()
```

> **Tip:** Define Gren classes in importable modules (not `__main__`); the artifact namespace is derived from the class's module + qualified name.

## Core Concepts

### How Caching Works

Each `Gren` instance maps deterministically to a directory based on its config:

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

Gren objects compose via nested configs. Each dependency gets its own artifact folder:

```python
import gren

class Dataset(gren.Gren[str]):
    name: str = gren.chz.field(default="toy")

    def _create(self) -> str:
        (self.gren_dir / "data.txt").write_text("hello\nworld\n")
        return "ready"

    def _load(self) -> str:
        return (self.gren_dir / "data.txt").read_text()


class TrainTextModel(gren.Gren[str]):
    dataset: Dataset = gren.chz.field(default_factory=Dataset)

    def _create(self) -> str:
        data = self.dataset.load_or_create()  # Triggers Dataset cache
        (self.gren_dir / "model.txt").write_text(f"trained on:\n{data}")
        return "trained"

    def _load(self) -> str:
        return (self.gren_dir / "model.txt").read_text()
```

### Storage Structure

```
$GREN_PATH/
├── data/                         # Default storage (version_controlled=False)
│   └── <module>/<Class>/
│       └── <hash>/
│           ├── .gren/
│           │   ├── metadata.json # Config, git info, environment
│           │   ├── state.json    # Status and timestamps
│           │   ├── gren.log    # Captured logs
│           │   └── SUCCESS.json  # Marker file
│           └── <your outputs>    # Files from _create()
├── git/                          # For version_controlled=True
│   └── <same structure>
└── raw/                          # Shared directory for large files
```

## Features

### GrenList: Managing Experiment Collections

`GrenList` provides a collection interface for organizing related experiments:

```python
import gren

class MyExperiments(gren.GrenList[TrainModel]):
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
class ModelWithValidation(gren.Gren[Path]):
    checkpoint_name: str = "model.pt"

    def _validate(self) -> bool:
        # Return False to force re-computation
        ckpt = self.gren_dir / self.checkpoint_name
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
print(f"Hash: {obj._gren_hash}")
print(f"Dir: {obj.gren_dir}")
```

### Serialization

Gren objects can be serialized to/from dictionaries:

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
class LargeDataProcessor(gren.Gren[Path]):
    def _create(self) -> Path:
        # self.raw_dir is shared across all configs
        # Create a subfolder for isolation if needed
        my_raw = self.raw_dir / self._gren_hash
        my_raw.mkdir(exist_ok=True)
        
        large_file = my_raw / "huge_dataset.bin"
        # ... write large file ...
        return large_file
```

### Version-Controlled Storage

For artifacts that should be stored separately (e.g., checked into git):

```python
class VersionedConfig(gren.Gren[dict], version_controlled=True):
    # Stored under $GREN_PATH/git/ instead of $GREN_PATH/data/
    ...
```

## Logging

Gren installs stdlib `logging` handlers that capture logs to per-artifact files.

```python
import logging
import gren

log = logging.getLogger(__name__)

class MyPipeline(gren.Gren[str]):
    def _create(self) -> str:
        log.info("Starting computation...")  # Goes to gren.log
        log.debug("Debug details...")
        return "done"
```

### Console Output

By default, gren logs to console using Rich in a compact format:

```
HHMMSS file.py:line message
```

Gren emits status messages like:
```
load_or_create TrainModel abc123def (missing->create)
load_or_create TrainModel abc123def (success->load)
```

### Explicit Setup

```python
import gren

# Eagerly install logging handlers (optional, happens automatically)
gren.configure_logging()

# Get the gren logger
logger = gren.get_logger()
```

## Error Handling

```python
from gren import GrenComputeError, GrenWaitTimeout, GrenLockNotAcquired

try:
    result = obj.load_or_create()
except GrenComputeError as e:
    print(f"Computation failed: {e}")
    print(f"State file: {e.state_path}")
    print(f"Original error: {e.original_error}")
except GrenWaitTimeout:
    print("Timed out waiting for another process")
except GrenLockNotAcquired:
    print("Could not acquire lock")
```

## Submitit Integration

Run computations on SLURM clusters via [submitit](https://github.com/facebookincubator/submitit):

```python
import submitit
import gren

executor = submitit.AutoExecutor(folder="submitit_logs")
executor.update_parameters(
    timeout_min=60,
    slurm_partition="gpu",
    gpus_per_node=1,
)

# Submit job and return immediately
job = my_gren_obj.load_or_create(executor=executor)

# Job ID is tracked in .gren/state.json
print(job.job_id)
```

Gren handles preemption, requeuing, and state tracking automatically.

## Dashboard

The web dashboard provides experiment browsing, filtering, and dependency visualization.

### Running the Dashboard

```bash
# Full dashboard with React frontend
gren-dashboard serve

# Or with options
gren-dashboard serve --host 0.0.0.0 --port 8000 --reload

# API server only (no frontend)
gren-dashboard api
```

Or via Python:
```bash
python -m gren.dashboard serve
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
| `GREN_PATH` | `./data-gren/` | Base storage directory |
| `GREN_LOG_LEVEL` | `INFO` | Console verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `GREN_IGNORE_DIFF` | `false` | Skip embedding git diff in metadata |
| `GREN_POLL_INTERVAL_SECS` | `10` | Polling interval for queued/running jobs |
| `GREN_WAIT_LOG_EVERY_SECS` | `10` | Interval between "waiting" log messages |
| `GREN_STALE_AFTER_SECS` | `1800` | Consider running jobs stale after this duration |
| `GREN_LEASE_SECS` | `120` | Compute lock lease duration |
| `GREN_HEARTBEAT_SECS` | `lease/3` | Heartbeat interval for running jobs |
| `GREN_PREEMPT_MAX` | `5` | Maximum submitit requeues on preemption |
| `GREN_CANCELLED_IS_PREEMPTED` | `false` | Treat SLURM CANCELLED as preempted |
| `GREN_RICH_UNCAUGHT_TRACEBACKS` | `true` | Use Rich for exception formatting |

Local `.env` files are loaded automatically if `python-dotenv` is installed.

### Programmatic Configuration

```python
import gren
from pathlib import Path

# Set/get root directory
gren.set_gren_root(Path("/my/storage"))
root = gren.get_gren_root()

# Access config directly
gren.GREN_CONFIG.ignore_git_diff = True
gren.GREN_CONFIG.poll_interval = 5.0
```

### Class-Level Options

```python
class MyPipeline(gren.Gren[Path], version_controlled=True):
    _max_wait_time_sec = 3600.0  # Wait up to 1 hour (default: 600)
    ...
```

## Metadata

Each artifact records:

| Category | Fields |
|----------|--------|
| **Config** | `gren_python_def`, `gren_obj`, `gren_hash`, `gren_path` |
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
from gren import (
    # Core
    Gren,
    GrenList,
    GREN_CONFIG,
    
    # Configuration
    get_gren_root,
    set_gren_root,
    
    # Errors
    GrenError,
    GrenComputeError,
    GrenLockNotAcquired,
    GrenWaitTimeout,
    MISSING,
    
    # Serialization
    GrenSerializer,
    
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
