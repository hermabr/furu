# TODO

## Completed

- [x] Add support for building to wheel

## AI Agent Docs

- [ ] Add a file for ai agents that tells them that i never want to use typing.Optional or typing.Any in my codebase, but always actual types. Also tell it that I prefer to have happy path and crashing rather than trying to recover errors with try/catch.

## Code Quality

### Typing
- [ ] Remove all use of `typing.Any`
- [ ] Change `typing.Optional[T]` to `T | None`
- [ ] Removing `typing.List` and `typing.Dict` since Python 3.9+ supports built-in generics, such as `list[T]` and `dict[K, V]` over `List[T]` and `Dict[K, V]`

### API Design
- [ ] Rename `hexdigest` to a name that makes more sense (consider making it private)

### Error Handling
- [ ] Throw/assert/raise on unexpected behavior rather than manual handling. This means every time something unexpected happens it simply crashes and i never use try/catch or returning None to handle unexpected behavior.

## Testing

- [ ] Make e2e tests use actual data (run scripts to generate data in `data-huldra`)
- [ ] Make conftest tests use actual subclasses of Huldra instead of manual JSON objects to get actual realistic data that is updated
  - [ ] Create multiple experiments with dependencies

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

- [ ] List and filter experiments
  - Filter by host, runtime, date, etc.
  - Filter by a field in the config
- [ ] Rerun experiments from UI or via code snippet
- [ ] Create new experiments with different hyperparameters from the UI and get code snippet
- [ ] Support parameter sweeps
- [ ] Migration helper / show stale runs that are no longer valid

### Experiment Visualization

- [ ] DAG overview of experiments
  - Show full DAG based on existing experiments
  - Show full DAG based on code
  - Rich information: counts per node type, subclass groupings
  - Interactive: clicking a node highlights connected nodes
  - Show subclass relationships
- [ ] Experiment details view
  - Full config with collapsible sections
  - Click to navigate to child experiments
  - View all children of an experiment
- [ ] File viewer for artifacts (parquet, JSON, JSONL)
- [ ] View experiment logs
- [ ] Show which experiments are version controlled

### UI/UX

- [ ] General UI improvements
- [ ] Remove custom Tailwind colors
- [ ] Support making graphs/charts given a result file such as a json or parquet file (decide: Python vs React)
- [ ] Explore: discover all available runs/experiments in code (or via JSON manifest for reproducibility dashboard)
- [ ] Show all output files of an experiment
- [ ] Config diff view - Compare two experiments side-by-side
- [ ] Copy buttons - Copy hash, Python snippet, directory path to clipboard
- [ ] Auto-refresh toggle - Periodically refresh data
- [ ] Live log streaming - Tail logs in real-time
- [ ] Keyboard shortcuts - Navigation with VIM (j/k, /, etc.)

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
