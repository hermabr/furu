# Furu TODO

## Core object model
- [x] `class MyCls(Furu[int])` makes a frozen dataclass with named args
- [x] post_init for validation
- [x] don't allow users to call create directly (context variable + `__init_subclass__`)
- [x] furu config

## Hashing & schema
- [x] furu_hash
- [x] compute basic schema and schema hash
- [x] record current schema per object to detect when schema changed since last run
- [x] custom walk hooks: let users define how to walk an unsupported object (e.g. pydantic) for both hashing and schema (extensible `ArtifactSerializer` registry: subclass auto-register, `__furu_serializer__`, `Annotated[T, Serializer]`, or per-object `artifact_serializers`)

## Serialization (save & load results)
- [x] start with pickle
- [x] pytree-like strategy (most things in json, some in custom files)
    - [x] json-native scalars, lists, and dicts
    - [x] tuple, set, frozenset, pathlib.Path wrappers
    - [x] dataclass and pydantic model wrappers
    - [x] numpy array artifact codec
    - [x] polars dataframe artifact codec
    - [x] atomic result directory publish through temporary bundle rename
    - [x] cache hits load persisted results without recomputing
- [x] lazy saving/loading
- [x] let users register handlers/codecs
- [ ] more built-in codecs (think hard about which)
- [ ] infer codec automatically from bare type hints (`save_as` and `Annotated[T, Codec]` already work)

## Storage
- [x] per-object `_storage_path` override (same pattern as executor)

## Locking (file / compute)
- [x] auto make/delete the lock file
- [x] don't allow others to run while file is locked
- [x] heartbeat and waiting
- [x] allow other processes to wait for the worker before resuming their own work
- [x] use threading.Thread for heartbeat
- [ ] move to zig?
- [ ] database/redis backend

## Metadata
- [x] basic metadata
- [x] write running/completed metadata during create
- [x] load from artifact
- [ ] record git info
- [ ] record machine/host info and other most-relevant metadata; think deeply about what to include
- [ ] support time traveling to a previous experiment

## Logging
- [x] log when loading or creating object
- [x] record run logs in furu_dir
- [x] scoped file handler for each active furu run
- [ ] use logging consistently
- [ ] structured logging
- [ ] support multiple processes (e.g. torchrun with 8 tasks)
- [ ] events.log, or rename old logs when starting new runs, to make debugging easier

## Error handling
- [x] capture errors
- [x] make the errors informative
- [x] write traceback and locals to per-run error logs
- [ ] use rich tracebacks

## Reproducibility / runtime code tracing
- [ ] trace code at runtime to find all functions and save/hash their ast
- [ ] sandbox the code
- [ ] detect changes in local files
- [ ] detect library versions (only of libraries we import?)
- [ ] detect all functions we call
- [ ] do not allow create to depend on/be affected by sys.argv
- [ ] take inspiration from pytest-cov
- [ ] use this information to invalidate loads

## Migration
- [x] support registering and doing basic migrations
- [ ] reject ambiguous migration graphs where more than one path connects the same source and target schema
- [ ] allow migration schemas to be provided as either the schema dict or a schema hash string
- [ ] support an explicit "unsupported" migration result for transforms that cannot migrate a matching source
- [ ] verify migrated runtime result links in status/create/load_existing before treating them as completed/cache hits (`is_migrated` exists; validation still incomplete)
- [ ] handle stale migration links whose source artifact or old schema directory was deleted

## Executor & dependencies
- [x] local create execution
- [x] local batched create_batched execution
- [x] slurm workers
- [x] eager dependencies declared with `@furu.dependency`
- [x] lazy dependencies captured when `.create()` is called inside a create fn (e.g. variable number of chunks)
- [ ] slurm dag
- [ ] time traveling executor (git worktrees?)
- [ ] tui for execution coordinator

## Tooling & interfaces
- [ ] querying/filtering
- [ ] sync support: move experiments/objects easily from one host to another
- [ ] dashboard
- [ ] make docs

## Testing
- [x] pytest plugin

## Future / misc
- [ ] decide if it's possible to inject information into a class (e.g. unknown sentences + their translations for leap)
- [ ] have a method for making the raw data so we have some sort of tracking
- [ ] decide if I have too many cached_properties
- [ ] decide what should be property vs method
- [ ] make sure the public api has very good names, with a clear flow of features
