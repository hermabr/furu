# TODO

## Completed

- [x] Add support for building to wheel

## Code Quality

### Typing
- [ ] Remove all use of `typing.Any`
- [ ] Change `typing.Optional[T]` to `T | None`

### API Design
- [ ] Rename `hexdigest` to a name that makes more sense (consider making it private)

### Error Handling

- [ ] Throw/assert/raise on unexpected behavior rather than manual handling. This means every time something unexpected happens it simply crashes and i never use try/catch or returning None to handle unexpected behavior.

## Testing

- [ ] Make e2e tests use actual data (run scripts to generate data in `data-huldra`)
- [ ] Make conftest tests use actual subclasses of Huldra instead of manual JSON objects to get actual realistic data that is updated
  - [ ] Create multiple experiments with dependencies

## Dashboard Features

### Experiment Management
- [ ] List and filter experiments
  - Filter by host, runtime, date, etc.
- [ ] Rerun experiments from UI or via code snippet
- [ ] Create new experiments with different hyperparameters
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
- [ ] Support making graphs/charts (decide: Python vs React)
- [ ] Explore: discover all available runs/experiments in code (or via JSON manifest for reproducibility dashboard)

## Build & Packaging

- [ ] Verify dashboard access: `uv add huldra` vs `uv add huldra[dashboard]`
- [ ] Consider moving from hatchling to uv-build
- [ ] Decide if `dashboard-frontend` should be inside `src/`
