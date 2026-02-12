# TODO

## General

- [ ] support doing time machine to go back to the state at the time an artifact was created
- [x] add a flag which makes one item always rerun, such as FURU_ALWAYS_RERUN="mypkg.my_file.MyObject"
- [x] Change the version controlled flow so that it is always saved in the current folder at the same level as the pyproject.toml if that exists and in .gitignore if that exists and if not it throws. This should happen even if the general furu directory is somewhere else. It should save it to something like furu-data/artifacts or a better similar name. It should be possible to override this with an env variable.
- [x] When waiting, say how long you will be waiting and how long since file was touched
- [?] Sometimes it gets stuck in waiting for compute lock forever
- [x] Rename to furu
- [x] Don't allow `<locals>` in `__qualname__` if not providing env flag
- [x] Verify that all tests including e2e and dashboard tests are using tmp directory for furu root data dir
- [x] make dashboard-dev should populate dummy data
- [ ] Better and more complex filtering, maybe inspired by wandb etc
- [ ] Flag for allowing running in either debug mode or running __main__. Probably it should redirect the output to a temp directory
- [ ] Option to add a _name or _names property which returns pretty names for my experiments. Maybe i want something similar to collections and tags too
- [ ] In general, if you first run _create i don't want to unnecessarily call _load again inside a .get. Also if there are other similar obvious places to make it faster, use those
    - I probably still want to reload in .get when i call it from another object, rather than caching it in memory in case the next task wants to reload it
- [ ] consider moving from time.time() to time.perf_counter() for measuring elapsed time
- [ ] Make a good system for versioning objects, maybe also make this work with the migration so that you can migrate from one version of an furu object to a newer version using migrations
- [ ] Add option to migrate everything that can be auto migrated where there already exist default values
- [ ] For migrations: Decide if its okay with two aliases from same base object with same schema
- [x] Rename `furu_version` to something more reasonable, such as `furu_package_version`
- [ ] Better filtering in the dashboard/code
- [ ] Add the high-level goals of furu
    - [ ] Keep everything in python, no bash, no yaml etc
    - [ ] Make everything reproducible without needing to think about it
    - [ ] Cache everything
        - [ ] Easily recompute things when you change them
    - [ ] Make it easy to reason about your code and to swap out elements
        - [ ] Allow model to easily use different datasets to train
- [ ] make nice method for finding all experiments or all successful experiments
- [ ] easy to load an object from a fururef
- [ ] throw error if i call .from_dict and it does not work
- [ ] rename window size to something better and more reasonable for the executors
- [ ] Don't fail if a dependency is failed, but rather resubmit it for submit slurm dag
    - [ ] Warn/throw/require flag if trying to submit more than 100 jobs
- [ ] Merge stuff from local worker and slurm pool together
- [x] better discovery of all existing (successful) experiments for a config
    - [ ] allow finding subclasses/parents too
- [ ] make furu_dep and furu_lazy_dep/furu_dynamic_dep, where furu_lazy_dep is for dependencies that are only known at runtime and not at build time
- [x] make tests/test_slurm_pool.py faster
- [ ] submit slurm workers as using batch for array job
- [x] add a load method that crashes if the object doesn't exist or change .get
- [ ] ensure import furu is fast

## Code Quality

- [ ] Make speed benchmarks and make operations faster, such as very large objects for hashing
- [ ] On errors, such as "you cannot run this from __main__", consider adding a comment telling it which env flag it can change (do this for all errors where we have flags and include in agents.md)
- [ ] Make the scanner faster/using cache
- [ ] Use cache where reasonable

## Storage & Data Management

- [ ] Garbage collection - Have the option to delete old, failed, or orphaned artifacts
- [ ] Disk usage tracking - Show storage consumption per namespace/experiment
- [ ] Export/import experiments - Package experiments with dependencies for sharing (the config/python code)

## Dependency & Cache Management

- [x] Reverse dependency tracking - Find which experiments depend on a given artifact
- [ ] Cascade invalidation - Option to invalidate downstream dependents when parent changes
- [ ] Orphan detection - Find artifacts no longer referenced by code
- [ ] Cache miss explanation - "This will recompute because field X changed"
- [ ] Hash diff tool - Show which fields differ between two experiments
- [x] check how to make implicit dependencies (probably this is doable with chz?)
    - [x] Add support for lazily computing dependencies (maybe)
    - [x] Have a def _dependencies(self) -> list[type[Furu]] which returns all the dependencies of the experiment that i don't want to define as fields
- [x] Define dependencies with a @furu_dep rather than with def _dependencies
- [ ] Check all the dependencies of an experiment both immediately and right before making an object if using an executor
    - [ ] Decide if this should be using a @furu_dep and @furu_lazy_dep or if it should always be done

## Execution & Compute

- [ ] Performance
    - [x] One option for recording git: FURU_RECORD_GIT="ignore|cached|uncached" (cached is default).
        - [x] Include git-remote opt-out with FURU_ALLOW_NO_GIT_ORIGIN.
    - [ ] Only make the furu_dir once
    - [ ] Decide everywhere I might want to use cached_property/cache
    - [ ] Make actual benchmarking tests
        - [ ] For general performance and what takes the most time
        - [ ] For how many writes/reads from disk i do per artifact i create
        - [ ] For how inefficient the executors are, both w.r.t. slurm and local executor
    - [ ] don't reload after create finishes in .get, but use the value you return in ._create
- [x] Always rerun flag (`FURU_ALWAYS_RERUN`) - Recompute even if artifact exists
- [ ] Resource tracking - Track peak memory, CPU time, GPU usage during `_create()`
- [?] Record how long it took to compute an artifact and show in the dashboard
- [ ] Add either .load or .load_or_throw which loads an artifact if it exists and otherwise throws an error
- [ ] support a .debug or .debug_run or .run_debug which will load all dependencies from where they usually get loaded, but that makes all new artifacts in some tmp directory (or env variable)
- [ ] Executor. Checklist:
    - [ ] I don't want backwards compatibility with any preserving of the old load_or_create, such as a _get_interactive method
    - [ ] Rather than having it return "cpu" as default spec, it should return "default" which you then provide to the executor in the `run` method
    - [ ] Remove the getattr if possible
    - [ ] Warn if calling .get on a node that is not in the dependencies
    - [ ] Don't call the new executor api a new package (currntly in 1.2 it says new package)
    - [ ] spec should be SlurmSpec | None, where None means default, rather than having a string
        - [ ] it might be fine to keep it as-is since we can verify this while building the executor. in that case, enforce this strictly that spec dict has all the keys
        - [ ] i should still support giving it some type safe thing
    - [ ] move the if self.exsits() and then load to very early (and maybe also speed it up even more after benchmarking, such as checking if .success file exists)
    - [ ] handle something was running but then later it crashes
    - [ ] Consider expanding build_plan() to expand dependencies for IN_PROGRESS/FAILED nodes when used for diagnostics/inspection (not just scheduling)
    - [ ] decide if there is a better name than force for .get that is more descriptive than force
    - [ ] rename window_size and "dfs"/"bfs" to something better
    - [ ] support batching for slurm job submissions (both for the per-job and for the mass-parallel)
    - [ ] in the per-job submissions, the slurm logs should be inside the <furu-hash>/.furu/submitit folder
    - [ ] decide if i even want the per-job submission (i probably do)
    - [ ] move from the current file system to sqlite (maybe)?
    - [ ] decide if i need "obj": { ...FuruSerializer dict... } inside the mass-parallel jobs
    - [ ] currently int he file system, it makes a runs/<run-id> folder where it puts the jobs. say i later submit a different run with some shared dependencies, how are the pending/currently running runs handled? does it periodically check if the runs are completed?
    - [ ] support one executor inheriting an alive worker in the mass-parallel world so that each executor doesn't need to spin up a new worker if one already exists and if the original executor no longer needs it
    - [ ] have periodic check in on running jobs in the mass-parallel world both by checking in the file system if they are .success and by sometimes (less often) querying slurm
    - [ ] decide on the rule for selecting which workers to spawn
        - [ ] based on which workers don't exist at all
        - [ ] based on estimated time to complete each
        - [ ] based on which are higher up in the tree
        - [ ] based on randomness
        - [ ] based on size of backlog
        - [ ] based on some rank over the specs that the user defines themselves
        - [ ] based on root node priority?
    - [ ] put the todos within a single spec in some ranked order (if stable, do something like namespace priority, class name priority)
        - [ ] maybe also the root node priority as the first entry?
    - [ ] for local, rather than having number of parallel jobs, have resources required by each job, for instance only one gpu job, but can have multiple cpu jobs in parallel
    - [ ] dashboard for experiment runner
    - [ ] the executor runner should also write out when it starts running a task
- [ ] compute lock heartbeat - check in on the queued items on occasion
- [ ] make the behavior for queuing better, where it checks if the processes that has queued it/the job is still alive

### Submitit

- [ ] Checkpointing - Resume long-running computations from checkpoints for slurm jobs
- [ ] Clean setup including dependencies for submitit jobs and simple config for setting up configs per class

## Dashboard Features

### Experiment Management

- [ ] Migration helper / show stale runs that are no longer valid
    - [x] Support migrations in the backend
    - [x] Migrate only one experiment
    - [ ] Migrate all to the new default value
    - [ ] Migrate all to a value you set
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
- [ ] Build static overview which produces a zip with with information about all experiments, but not the large files. This will allow us to call one function on the remote machine, download a zip and then open the dashboard locally to get a full overview of current state of experiments

### UI/UX

- [x] Use shadcn
- [ ] Use polars filtering for selecting experiments (there probably exists something better than polars)
- [ ] Support making graphs/charts given a result file such as a json or parquet file (decide: Python vs React)
- [x] Explore: discover all available runs/experiments in code (or via JSON manifest for reproducibility dashboard)
- [ ] Show all output files of an experiment
- [ ] Config diff view - Compare two experiments side-by-side
- [ ] Copy buttons - Copy hash, Python snippet, directory path to clipboard
- [ ] Auto-refresh toggle - Periodically refresh data
- [ ] Live log streaming - Tail logs in real-time
- [ ] Keyboard shortcuts - Navigation with VIM (j/k, /, etc.)
- [ ] General UI improvements
- [ ] Named experiments (either with _name in furu or rename in the web ui and update the metadata)
- [ ] Tags for experiments. Each experiment can have multiple tags
- [ ] Nice UI/UX for selecting experiments (maybe using either something from SQL or some sort of code for selecting?)
- [ ] Add terminal dashboard (tui) using opentui

## Documentation

- [ ] API reference docs - Auto-generated from docstrings
- [ ] Tutorial/quickstart guide - Beyond the examples
- [ ] Architecture overview - How the pieces fit together
- [ ] Changelog - Track breaking changes
- [ ] make good nested documentation/skills for how to use furu
    - [ ] example pipeline and best practices, such as subclassing, bumping version etc
    - [ ] how to use the dashboard
    - [ ] low level design for furu and how it works
    - [ ] what functionality/api do you have when using furu

## Build & Packaging

- [x] Consider moving from hatchling to uv-build
- [x] Add CI workflow to run tests on every push/PR
- [x] Add support for building to wheel

## Research & Investigation

- [ ] Understand what "absent" status means
