# Examples

These examples are meant to be run from the repo root.

They set `FURU_PATH` programmatically to `examples/.furu/` so you don’t clutter your working directory.
They also skip recording git metadata (equivalent to `FURU_RECORD_GIT=ignore`), so the examples work even if your working tree is large.

## Run

```bash
uv run python examples/run_train.py
uv run python examples/run_nested.py
uv run python examples/run_logging.py
uv run python examples/run_local_executor_benchmark.py
```

The local executor benchmark builds a nested DAG with 10,000 objects by default
and reports:

- `build_plan()` latency before execution
- end-to-end `run_local()` runtime with planner time share
- `build_plan()` latency after execution and warm-cache `run_local()` runtime

## Outputs

Artifacts will be written under:

- `examples/.furu/data/...`
- logs under each artifact directory: `.../.furu/furu.log`
