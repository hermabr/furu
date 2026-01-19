# TODO

## Completed

- [x] Add support for building to wheel

## General

- [ ] make good nested documentation/skills for how to use huldra
    - [ ] example pipeline and best practices, such as subclassing, bumping version etc
    - [ ] how to use the dashboard
    - [ ] low level design for huldra and how it works
    - [ ] what functionality/api do you have when using huldra
- [ ] support doing time machine to go back to the state at the time an artifact was created
- [ ] check how to make implicit dependencies (probably this is doable with chz?)
- [ ] add a flag which makes one item always rerun, such as HULDRA_ALWAYS_RERUN="mypkg.my_file.MyObject"
- [ ] Add support for lazily computing dependencies (maybe)
- [ ] Change the version controlled flow so that it is always saved in the current folder at the same level as the pyproject.toml if that exists and in .gitignore if that exists and if not it throws. This should happen even if the general gren directory is somewhere else. It should save it to something like gren-data/artifacts or a better similar name. It should be possible to override this with an env variable.
- [ ] When waiting, say how long you will be waiting and how long since file was touched
- [ ] Sometimes it gets stuck in waiting for compute lock forever
- [ ] Rename to furu
- [ ] Add terminal dashboard (tui)

## Code Quality

- [ ] Make speed benchmarks and make operations faster, such as very large objects for hashing
- [ ] On errors, such as "you cannot run this from __main__", consider adding a comment telling it which env flag it can change (do this for all errors where we have flags and include in agents.md)

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

- [ ] Dry-run mode (`GREN_DRY_RUN`) - Preview what would be computed without running
- [x] Force recompute flag (`GREN_FORCE_RECOMPUTE`) - Recompute even if artifact exists
- [ ] Checkpointing - Resume long-running computations from checkpoints
- [ ] Resource tracking - Track peak memory, CPU time, GPU usage during `_create()`

## Dashboard Features

### Experiment Management

- [ ] Migration helper / show stale runs that are no longer valid
    - [ ] Support migrations in the backend
    - [ ] Migrate all to the new default value
    - [ ] Migrate all to a value you set
    - [ ] Migrate only one experiment
    - [ ] Migrate based on a filter
- [ ] Create new experiments with different hyperparameters from the UI and get code snippet
- [ ] Support parameter sweeps
- [ ] Rerun experiments from UI or via code snippet
- [ ] Make helpers, such as auto complete and a nice helper, such as option to first add one filter and then add another filter and also change between larger than/smaller than etc. For things like filtering for users, have a dropdown if there are few enough options. The autocomplete might also be better if it is for per-config rather than general for all of them.

### Experiment Visualization

- [ ] DAG overview of experiments
  - [x] Show full DAG based on existing experiments
  - [x] Interactive: clicking a node highlights connected nodes
  - [ ] Rich information: counts per node type, subclass groupings
  - [ ] There should be a button for in each node of the DAG to see all experiments of that dag
  - Show subclass relationships
  - Support either DAG made from all experiments or by crawling through the actual source code
    - [x] From experiments
    - [ ] From source code
- [x] Experiment details view
  - [ ] Full config with collapsible sections
  - [ ] Click to navigate to child experiments
  - [ ] View all children of an experiment
  - [ ] Show the config in a tree view and maybe have a DAG-like view so that you can see what config variables are for what node and which nodes depend on which
- [ ] File viewer for artifacts (parquet, JSON, JSONL)
- [ ] View experiment logs
- [ ] Show which experiments are version controlled
- [ ] Support different ways of sorting the experiments, such as by subclass, by time created, by updated time, by dependencies, by runtime, by status (e.g., running before queued)

### UI/UX

- [x] Use shadcn
- [ ] Use polars filtering for selecting experiments (there probably exists something better than polars)
- [ ] Support making graphs/charts given a result file such as a json or parquet file (decide: Python vs React)
- [ ] Explore: discover all available runs/experiments in code (or via JSON manifest for reproducibility dashboard)
- [ ] Show all output files of an experiment
- [ ] Config diff view - Compare two experiments side-by-side
- [ ] Copy buttons - Copy hash, Python snippet, directory path to clipboard
- [ ] Auto-refresh toggle - Periodically refresh data
- [ ] Live log streaming - Tail logs in real-time
- [ ] Keyboard shortcuts - Navigation with VIM (j/k, /, etc.)
- [ ] General UI improvements
- [ ] Named experiments (either with _name in gren or rename in the web ui and update the metadata)
- [ ] Tags for experiments. Each experiment can have multiple tags
- [ ] Nice UI/UX for selecting experiments (maybe using either something from SQL or some sort of code for selecting?)

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

- [ ] Explore if I the dashboard feature can be added in a different way, so that type checking works correctly for the main gren package, so that the normal package cannot use packages only available in the dashboard
- [ ] Consider moving from hatchling to uv-build
- [ ] Add CI workflow to run tests on every push/PR

## Research & Investigation

- [ ] Understand what "absent" status means
