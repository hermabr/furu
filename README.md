# huldra

NOTE: this is a very early prototype; expect breaking changes.

`huldra` is a small library for building cacheable, nested pipelines in Python.
You define a computation as a config object; `huldra` turns that config into a stable
on-disk artifact directory, records metadata/state, and reuses results across runs.

It is built on top of [chz](https://github.com/openai/chz) for declarative configs.

## Install

This repo is managed with `uv`:

```bash
uv add huldra
```

Or install as a package (once published / in your own environment):

```bash
pip install huldra
```

## Quickstart

1. Subclass `huldra.Huldra[T]`
2. Implement:

- `_create(self) -> T` (compute and write outputs into `self.huldra_dir`)
- `_load(self) -> T` (load outputs back from `self.huldra_dir`)

3. Call `load_or_create()`

Example (define the pipeline in an importable module, and call it from a separate script):

```python
# my_project/pipelines.py
import json
from pathlib import Path

import huldra


class TrainModel(huldra.Huldra[Path]):
    lr: float = huldra.chz.field(default=1e-3)
    steps: int = huldra.chz.field(default=1000)

    def _create(self) -> Path:
        # write outputs into the artifact directory
        (self.huldra_dir / "metrics.json").write_text(
            json.dumps({"lr": self.lr, "steps": self.steps})
        )
        ckpt = self.huldra_dir / "checkpoint.bin"
        ckpt.write_bytes(b"...")
        return ckpt

    def _load(self) -> Path:
        # load outputs back from disk
        return self.huldra_dir / "checkpoint.bin"
```

```python
# run_train.py
from my_project.pipelines import TrainModel

artifact = TrainModel(lr=3e-4, steps=5000).load_or_create()
print("artifact at:", artifact)
```

Tip: define Huldra classes in importable modules (not `__main__`); the artifact namespace is derived from the class’s module + qualified name.

## Nested pipelines (dependencies)

Huldra objects are meant to compose via nested config objects.
When one object calls `other.load_or_create()`, the dependency gets its own artifact folder.

```python
# my_project/pipelines.py
import logging

import huldra

log = logging.getLogger(__name__)


class Dataset(huldra.Huldra[str]):
    name: str = huldra.chz.field(default="toy")

    def _create(self) -> str:
        log.info("preparing dataset: %s", self.name)
        (self.huldra_dir / "data.txt").write_text("hello\nworld\n")
        return "ready"

    def _load(self) -> str:
        return (self.huldra_dir / "data.txt").read_text()


class TrainTextModel(huldra.Huldra[str]):
    dataset: Dataset = huldra.chz.field(default_factory=Dataset)

    def _create(self) -> str:
        log.info("training")
        data = self.dataset.load_or_create()
        (self.huldra_dir / "model.txt").write_text(f"trained on:\n{data}")
        return "trained"

    def _load(self) -> str:
        return (self.huldra_dir / "model.txt").read_text()
```

## Where artifacts go

Each `Huldra` instance maps deterministically to a directory:

- `huldra_dir = <root>/<namespace>/<hash>/`
- `<namespace>` is derived from the class’s module + qualified name
- `<hash>` is derived from the object’s config values

The root defaults to `./data-huldra/` and can be overridden:

- `HULDRA_PATH`: base storage directory (default: `data-huldra`)
- `HULDRA_IGNORE_DIFF`: set to `1/true/yes` to avoid embedding large git diffs in metadata

You can also point to a “version controlled” root by setting `version_controlled = True` on a subclass.

## Raw directory

Huldra also exposes a shared “raw” directory:

- `self.raw_dir == huldra.HULDRA_CONFIG.raw_dir` (defaults to `<HULDRA_PATH>/raw`)

Use this for large, non-versioned inputs/outputs. If you want per-object isolation, create a subfolder yourself, e.g. `self.raw_dir / self.hexdigest / "file.ext"`.

## Logging (Hydra-style)

Huldra installs stdlib `logging` handlers on the _root logger_:

- `current_holder.huldra_dir / ".huldra" / "huldra.log"` while a holder is active

This means you can use regular Python logging inside your code:

```python
import logging
log = logging.getLogger(__name__)
log.info("hello from my pipeline")
```

Huldra automatically tracks which holder is active during `load_or_create()`, so nested
dependencies log into their own folders and then logging reverts back to the parent.

By default, Huldra also logs to the console using Rich in a compact, single-line format:

- `HHMMSS file.py:line message` (no explicit `INFO`/`DEBUG` text)
- `file.py:line` is colored by level (debug/info/warn/error)

Huldra emits an INFO summary per call, e.g. `load_or_create <ClassName> <hash> (missing->create|running->wait|success->load)`.

Each `load_or_create()` call writes a separator line (`------------------`) to the per-artifact `huldra.log` file.

If you want to install the handler eagerly (optional), call:

```python
import huldra
huldra.configure_logging()
```

Control console verbosity with:

- `HULDRA_LOG_LEVEL` (default: `INFO`)

Note: `huldra.log` files remain detailed and include timestamps + `[LEVEL]` for auditing/debugging.

## Metadata and state

Each artifact directory includes files maintained by Huldra:

- `.huldra/metadata.json`: config, hash, directory, environment info, and git info (when available)
- `.huldra/state.json`: status transitions (`missing/queued/running/success/failed/preempted`) and timestamps

This is used to detect whether an object “exists” and to coordinate concurrent runs via lock files.

## Submitit integration

`load_or_create(executor=...)` supports running work via `submitit`.
Huldra stores submitit job handles in `<huldra_dir>/.huldra/` and tracks job IDs in `.huldra/state.json`.

## Configuration knobs

Environment variables:

- `HULDRA_PATH`: base storage directory (default: `data-huldra`)
- `HULDRA_POLL_INTERVAL_SECS`: polling interval for queued/running jobs
- `HULDRA_STALE_AFTER_SECS`: consider running jobs stale after this many seconds
- `HULDRA_PREEMPT_MAX`: max submitit requeues on preemption
- `HULDRA_CANCELLED_IS_PREEMPTED`: treat CANCELLED as preempted (default: false)

Local `.env` loading is supported if `python-dotenv` is installed.

## Features

- Cacheable computations (config → artifact directory)
- Nested dependency composition (pipelines as nested configs)
- Atomic state tracking and lock files for safe concurrent runs
- Metadata capture (environment + optional git details)
- Submitit integration for scheduled execution
- Scoped logging to the active artifact directory using stdlib `logging`

## Non-goals / caveats

- Prototype status: APIs and on-disk formats may change
- Not a workflow scheduler; it’s a lightweight caching layer for Python code
