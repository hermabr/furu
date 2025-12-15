# Examples

These examples are meant to be run from the repo root.

They set `HULDRA_PATH` programmatically to `examples/.huldra/` so you don’t clutter your working directory.
They also disable embedding the git diff in metadata (equivalent to `HULDRA_IGNORE_DIFF=1`), so the examples work even if your working tree is large.

## Run

```bash
uv run python examples/run_train.py
uv run python examples/run_nested.py
uv run python examples/run_logging.py
```

## Outputs

Artifacts will be written under:

- `examples/.huldra/data/...`
- logs under each artifact directory: `.../huldra.log`
