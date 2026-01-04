# TODO

## Completed

- [x] Add support for building to wheel

## Code Quality

## Storage & Data Management

- [ ] Garbage collection - Have the option to delete old, failed, or orphaned artifacts
- [ ] Disk usage tracking - Show storage consumption per namespace/experiment
- [ ] Export/import experiments - Package experiments with dependencies for sharing (the config/python code)

## Dependency & Cache Management

- [ ] Reverse dependency tracking - Find which experiments depend on a given artifact
- [ ] Cascade invalidation - Option to invalidate downstream dependents when parent changes
- [ ] Orphan detection - Find artifacts no longer referenced by code
- [ ] Cache miss explanation - "This will recompute because field X changed"
- [ ] Hash diff tool - Show which fields differ between two experiments

## Execution & Compute

- [ ] Dry-run mode (`HULDRA_DRY_RUN`) - Preview what would be computed without running
- [ ] Force recompute flag (`HULDRA_FORCE_RECOMPUTE`) - Recompute even if artifact exists
- [ ] Checkpointing - Resume long-running computations from checkpoints
- [ ] Resource tracking - Track peak memory, CPU time, GPU usage during `_create()`

## Dashboard Features

### Experiment Management

- [ ] Migration helper / show stale runs that are no longer valid
- [ ] Create new experiments with different hyperparameters from the UI and get code snippet
- [ ] Support parameter sweeps
- [ ] Rerun experiments from UI or via code snippet
- [ ] Make helpers, such as auto complete and a nice helper, such as option to first add one filter and then add another filter and also change between larger than/smaller than etc. For things like filtering for users, have a dropdown if there are few enough options. The autocomplete might also be better if it is for per-config rather than general for all of them.

### Experiment Visualization

- [ ] DAG overview of experiments
  - [ ] Show full DAG based on existing experiments
  - [ ] Rich information: counts per node type, subclass groupings
  - [ ] Interactive: clicking a node highlights connected nodes
  - Show subclass relationships
  - Show full DAG based on code
- [ ] Experiment details view
  - Full config with collapsible sections
  - Click to navigate to child experiments
  - View all children of an experiment
- [ ] File viewer for artifacts (parquet, JSON, JSONL)
- [ ] View experiment logs
- [ ] Show which experiments are version controlled

### UI/UX

- [ ] Use shadcn
- [ ] Support making graphs/charts given a result file such as a json or parquet file (decide: Python vs React)
- [ ] Explore: discover all available runs/experiments in code (or via JSON manifest for reproducibility dashboard)
- [ ] Show all output files of an experiment
- [ ] Config diff view - Compare two experiments side-by-side
- [ ] Copy buttons - Copy hash, Python snippet, directory path to clipboard
- [ ] Auto-refresh toggle - Periodically refresh data
- [ ] Live log streaming - Tail logs in real-time
- [ ] Keyboard shortcuts - Navigation with VIM (j/k, /, etc.)
- [ ] General UI improvements

### API (Missing Endpoints)

- [ ] `GET /api/experiments/{id}/logs` - Return log file contents
- [ ] `GET /api/experiments/{id}/artifacts` - List files in artifact directory
- [ ] `DELETE /api/experiments/{id}` - Delete an experiment
- [ ] `POST /api/experiments/{id}/invalidate` - Invalidate a cached result
- [ ] `GET /api/namespaces` - List all unique namespaces for filtering

## Documentation

- [ ] API reference docs - Auto-generated from docstrings
- [ ] Tutorial/quickstart guide - Beyond the examples
- [ ] Architecture overview - How the pieces fit together
- [ ] Changelog - Track breaking changes

## Build & Packaging

- [ ] Explore if I the dashboard feature can be added in a different way, so that type checking works correctly for the main huldra package, so that the normal package cannot use packages only available in the dashboard
- [ ] Consider moving from hatchling to uv-build

## Research & Investigation

- [ ] Understand what "absent" status means
