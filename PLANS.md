# ExecPlan: Batch-aware `load_or_create` with manifest-backed multi-locks

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository did not include a checked-in `PLANS.md` when this task started, so this file is the source of truth for the task. Keep it self-contained. Do not assume the reader has the design PDF, this chat, or any prior plan.

## Purpose / Big Picture

After this change, Furu users can keep the simple per-object storage model they already have while asking for several artifacts in one call. A user will be able to write `furu.load_or_create([obj1, obj2, obj1])` and get one result per input entry in the original order, even though the engine computes only the unresolved unique objects. Concrete Furu classes will express creation in exactly one place: either `_create` for single-item creation or `_create_batched` for class-specific batched creation. The storage layout must stay unchanged: one artifact directory, one `result.pkl`, one `metadata.json`, and one `run.log` per object.

The visible proof is behavioral, not architectural. New tests must show that single-object calls still work, single-object calls on batch-only classes work through the shared engine, list calls deduplicate by cache identity, already-computed objects are skipped before locking and re-checked after locking, mixed concrete classes are partitioned internally but still handled by one public call, and overlapping batch executions do not duplicate work for shared unresolved objects.

## Progress

- [x] (2026-04-03 00:00Z) Drafted a repository-specific ExecPlan from the current working tree and the batching design report.
- [x] (2026-04-03 07:20Z) Read the current core, logging, locking, validation, export surface, README, and the main test files to anchor the implementation against the existing behavior.
- [x] (2026-04-03) Implemented the public batch-aware API and creation-hook validation in `src/furu/core.py`, `src/furu/validate.py`, and `src/furu/__init__.py`.
- [x] (2026-04-03) Refactored execution into shared helpers in `src/furu/core.py`, including deduplication, pre-lock skipping, post-lock re-checking, batch grouping, per-object persistence, and failure handling.
- [x] (2026-04-03) Extended scoped logging so batch compute can write to each participating object's `run.log` without losing nested child scoping.
- [x] (2026-04-03) Replaced the single-path lock implementation with manifest-backed `lock_many(...)` and kept `lock(...)` as a one-path specialization.
- [x] (2026-04-03) Added and updated tests in `tests/test_core.py`, `tests/test_locking.py`, and `tests/test_furu_locking_contention.py`.
- [x] (2026-04-03) Ran `uv run pytest -q`, `uv run ruff check`, and `uv run ty check` successfully after the final code changes.

## Surprises & Discoveries

- Observation: `src/furu/core.py` currently puts the entire `load_or_create` implementation inside the instance method on `Furu`, so there is no top-level executor to reuse for list calls.
  Evidence: `Furu.load_or_create` currently contains cache-hit checks, locking, metadata writes, `_create()` execution, result persistence, and error logging in one method.

- Observation: `src/furu/core.py` makes `_create` an abstract method, which means batch-only subclasses cannot exist without changing class validation.
  Evidence: `Furu` inherits from `ABC`, and `_create` is marked with `@abstractmethod`.

- Observation: `src/furu/logging.py` scopes file logging to a single active path via one context variable, so `_create_batched` would otherwise log only to the fallback file or to one arbitrary object's log.
  Evidence: `_CURRENT_LOG_PATH` stores a single `Path | None`, and `_ScopedFileHandler.emit` opens exactly one log file per record.

- Observation: `src/furu/locking.py` stores only the claim path string in the lock inode today, which is enough for a single visible lock name but not enough to break a stale lock group safely.
  Evidence: `lock(...)` writes `str(owner_claim_path)` into the claim file, and stale breaking reads that same string back through `_safe_read_breakable_claim_path`.

- Observation: several existing locking tests inspect lock file contents directly and assume the old raw-claim format.
  Evidence: `tests/test_locking.py` and `tests/test_furu_locking_contention.py` read the lock path text, create ad hoc claim files, and compare claim-path strings.

- Observation: the repository had no plan file on disk, so making the ExecPlan a true living document required creating one before implementation.
  Evidence: `rg --files` at repository root showed no `PLANS.md` or other checked-in ExecPlan file.

- Observation: the same-thread reentry guard had to wrap lock acquisition, not just the compute step, or recursive unresolved loads could still block on a lock already owned by the same thread.
  Evidence: a recursive `load_or_create()` path could enter `lock_many(...)` before reaching the original guard placement, so the guard was moved to cover the lock-and-recheck section.

- Observation: stale-lock tests that relied on live child processes were timing-sensitive because the heartbeat refreshes the same inode that the stale check inspects.
  Evidence: a full-suite run exposed a flaky stale-break test; rewriting the stale-lock tests to create stale manifests directly removed the race.

- Observation: `ty` rejects concrete `_create_batched(...)` overrides typed as `list[Self]` against the shared base method, even though the runtime always passes an ordered list.
  Evidence: `uv run ty check` reported `invalid-method-override` until the checked-in hook signature was widened to `Sequence[...]` for static typing while the executor continued to pass lists.

## Decision Log

- Decision: Keep batching as an execution optimization only. Do not introduce a new on-disk batch artifact or change result storage paths.
  Rationale: The current mental model is per-object caching. The design goal is faster execution for shared upstream work, not a second storage system.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: Make top-level `furu.load_or_create(...)` the canonical executor and turn `Furu.load_or_create(...)` into a thin wrapper.
  Rationale: There must be exactly one execution pipeline, or single and list calls will drift over time.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: A concrete subclass may define at most one of `_create` and `_create_batched` in its class body. If it defines neither, it must inherit an already-resolved mode from a base class; otherwise class creation fails.
  Rationale: This preserves the “express creation in one place” rule while keeping inheritance usable.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: Support mixed concrete classes at runtime, but keep the public overloads simple and primarily homogeneous for type checking.
  Rationale: Users should be able to pass one list and let the engine partition internally, even though Python's static typing cannot model every heterogeneous case elegantly.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: Extend file logging to support multiple simultaneous scoped log paths during batched compute, and keep nested single-object scopes authoritative when a child object calls `load_or_create()` inside that batch.
  Rationale: The storage invariant still promises one `run.log` per object, and nested dependencies must continue to log to their own files rather than leaking into the parent or batch logs.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: Write all new locks in manifest format and teach stale-lock parsing to understand both the new JSON manifest and the legacy one-path claim-string format.
  Rationale: Alpha status means strict compatibility is not required, but tolerant stale breaking prevents old stale locks from trapping the new implementation.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: `lock_many(...)` must refuse lock sets that span multiple filesystems rather than silently falling back to a weaker mechanism.
  Rationale: Hardlinks are a core correctness property of the current design, not an optional optimization.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: An empty sequence passed to top-level `load_or_create(...)` returns an empty list immediately.
  Rationale: It is the least surprising behavior and avoids special-case caller code.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: Add a lightweight same-thread self-reentry guard for obvious “compute the same unresolved object while already computing it” cases.
  Rationale: The design report calls this out as a likely deadlock footgun. A small guard buys better errors without changing normal dependency loading.
  Date/Author: 2026-04-03 / ChatGPT

- Decision: Type the shared `_create_batched(...)` hook as accepting a `Sequence[...]` even though the executor passes a concrete `list`.
  Rationale: This keeps the runtime behavior unchanged while satisfying the repository's `ty` checker for concrete subclass overrides.
  Date/Author: 2026-04-03 / ChatGPT

## Outcomes & Retrospective

The repository now has one shared top-level `load_or_create(...)` executor, manifest-backed `lock_many(...)`, multi-target scoped logging, early create-hook validation, updated README usage docs, and no storage migration. The behavior is covered by expanded core, locking, and contention tests, including duplicate preservation, pre-lock skipping, post-lock rechecks, mixed-class partitioning, batch log fanout, whole-group stale breaking, and overlapping batch requests.

Final validation succeeded with:

- `uv run pytest -q`
- `uv run ruff check`
- `uv run ty check`

## Context and Orientation

The repository is small and centered in `src/furu/`. `src/furu/core.py` defines `Furu`, computes `data_dir`, owns the current `load_or_create` lifecycle, and persists `result.pkl`, `.furu/metadata.json`, and `.furu/run.log`. `src/furu/locking.py` implements the current hardlink lock with heartbeat, stale breaking, and `LockLostError` handling. `src/furu/logging.py` uses a context variable to decide which file receives logs. `src/furu/validate.py` wires `@validate` methods into `__post_init__`, and `src/furu/__init__.py` is the public export surface. The test suite is concentrated in `tests/test_core.py`, `tests/test_locking.py`, `tests/test_furu_locking_contention.py`, and `tests/test_errors.py`.

In this plan, “cache identity” means the normalized `data_dir` path for a Furu object. Two Python objects that resolve to the same `data_dir` are duplicates for compute and locking purposes, even if they are different instances. “Unresolved” means the result file does not exist at the time the executor plans work. “Pending” means an object was unresolved before locking and is still unresolved after the post-lock re-check. “Batch group” means the ordered pending objects that share the same concrete class and therefore the same creation hook. “Manifest-backed lock” means a hardlinked lock whose inode contains JSON naming the claim file and every visible lock path in that lock set.

The repository’s current `AGENTS.md` adds two important constraints. Use `uv run ruff check` and `uv run ty check` for static validation, and do not preserve backward compatibility for its own sake because the package is still alpha. Respect both rules during implementation.

## Plan of Work

Start in `src/furu/core.py` by extracting the current single-object `load_or_create` body into reusable helpers and adding a new top-level overloaded `load_or_create(...)`. The instance method on `Furu` should become a one-line delegation to the top-level function. The new top-level executor must accept either one object or a sequence, normalize to a list, deduplicate by canonical `data_dir`, skip already-computed items before locking, create `.furu` directories for still-missing items, acquire one multi-lock over the unresolved unique objects when `use_lock=True`, re-check existence after the lock is held, and then execute compute only for the remaining pending objects. Return materialization must preserve the original input length and order and repeat the same loaded or computed value when the input contained duplicates.

While refactoring `src/furu/core.py`, add a clear creation-mode resolver instead of trying to make `ABC` express “one of two hooks”. Remove the abstract `_create` requirement, define a stub `_create_batched` on `Furu`, and add validation that resolves each concrete subclass to either single or batched mode. Keep the inheritance rule explicit: the current class body may define one hook or neither, but never both. If the class defines neither and no base class has already resolved the mode, fail class creation with a crisp `TypeError`. The executor should branch by resolved mode for each batch group: single-mode classes run `_create()` sequentially under the batch lock, and batched-mode classes call `_create_batched(group)` exactly once and must return one result per input object in the same order.

Refactor persistence and failure handling next. Factor out helpers that write `RunningMetadata`, persist one result atomically with the same temp-file-and-rename pattern the code uses today, load cached results from disk, and write the current rich error log format into `.furu/error-*.log`. Keep persistence per-object and non-transactional across the batch. If a batch hook returns ten results and persisting the fourth fails, the first three objects should stay completed and the remaining ones should stay unresolved. If a batch hook raises before any result is returned, write equivalent failure traces for each object in that group and re-raise the original exception.

Update `src/furu/logging.py` before finalizing batch execution. Replace the single active log path with an ordered tuple of active log paths and add `_scoped_log_files(...)`. During batched compute for one group, scope logs to that group’s `run.log` files so the class’s `_create_batched(...)` messages land in every participating artifact log. When persisting or loading one specific object, narrow the scope back to that object’s file so object-specific persistence messages do not leak into unrelated logs. Nested calls to `load_or_create()` must still override the current scope and log only to the nested object’s log file, exactly like the existing parent-child logging behavior.

Then move to `src/furu/locking.py`. Introduce `lock_many(lock_paths, *, lifetime_s=..., heartbeat_interval_s=..., acquire_timeout_s=..., acquire_poll_interval_s=...)` as the new primitive and re-implement `lock(lock_path, ...)` as `lock_many([lock_path], ...)`. Normalize the input paths by resolving, sorting, and deduplicating them. Verify they all live on the same filesystem device. Create one unique claim file, write a JSON manifest containing `version`, `claim_path`, and the full ordered `lock_paths`, and touch the claim inode into the future exactly as the current code does. Acquire visible lock names in sorted order by hardlinking the shared claim inode into each target path. If acquisition fails at any path, release the subset already acquired during that attempt, optionally break a stale group if the failed path is stale, sleep based on the failed path’s expiry, and retry from the top. The `has_lock()` callback must return `True` only if every visible lock path still points to the same inode as the claim file.

Change stale-lock breaking from a one-path cleanup to a whole-group cleanup. Replace the current “claim path string” parser with a manifest reader that can normalize both the new JSON format and the current legacy single-path text format. Use that normalized manifest to discover the entire lock set. When one visible lock path looks stale, coordinate stale breaking using a break directory derived from the claim path so two breakers encountering different member paths of the same stale batch still contend on the same stale-break guard. Once the breaker wins, verify that the observed visible lock path still belongs to the same owner inode, re-check staleness, then unlink every listed visible lock path that still matches that inode and finally remove the claim path. Preserve the current error semantics for unexpected stat, link, or unlink errors.

Finally, update the tests and public surface. Export the new top-level `load_or_create` from `src/furu/__init__.py`. Add focused core tests for list execution, batch-only classes, duplicate preservation, pre-existing-item skipping, mixed-class partitioning, invalid hook definitions, inheritance rules, same-thread self-reentry, and batch log scoping. Add locking tests for `lock_many(...)`, manifest parsing, `has_lock()` failure when any member path is lost, whole-group stale breaking, cross-device rejection, and the one-path wrapper behavior through `lock(...)`. Add at least one contention test showing that overlapping batch requests do not duplicate compute for the shared unresolved object set. Finish by updating the short public docs in `README.md` with a minimal example of the new top-level API and `_create_batched(...)` hook.

### Milestone 1: Public API and class-mode resolution

At the end of this milestone, the repository has a top-level `furu.load_or_create(...)`, the instance method delegates to it, and classes can resolve to either single or batched creation mode. No multi-locking is required yet, but the executor shape and hook-validation rules must already be in place. Implement the overloaded top-level function in `src/furu/core.py`, export it from `src/furu/__init__.py`, and add class-validation helpers in `src/furu/validate.py` or an adjacent helper in `src/furu/core.py`. Remove the `ABC`/`@abstractmethod` dependency that prevents batch-only classes.

Run `uv run pytest tests/test_core.py -q` after adding or updating core tests for invalid hook definitions, inheritance, empty-list behavior, single-object delegation, and list deduplication without batch-specific locking. Acceptance for this milestone is that a new batch-only test class can be instantiated, `obj.load_or_create()` on that class works by going through the shared engine, and the return-order invariant holds for duplicate inputs.

### Milestone 2: Shared executor, metadata, failure handling, and logging

At the end of this milestone, the shared executor can compute list inputs correctly under the old one-path lock or under `use_lock=False`, and all per-object file invariants still hold. Implement helper functions inside `src/furu/core.py` for normalization, deduplication, planning, grouping, metadata writing, result persistence, result loading, and failure logging. Update `src/furu/logging.py` so batch-scoped logs fan out to all participating `run.log` files without breaking nested child-object logging.

Run `uv run pytest tests/test_core.py -q` again after adding tests for batch hook call shape, skipping already-computed items, mixed-class grouping, non-transactional persistence, and log-file fanout. Acceptance for this milestone is that all new core tests pass and every participating object still gets exactly one artifact directory with one `result.pkl`, one `metadata.json`, and one `run.log`.

### Milestone 3: Manifest-backed `lock_many(...)`

At the end of this milestone, `src/furu/locking.py` exposes a general multi-lock context manager, and `lock(...)` is only a wrapper around it. Implement manifest writing, path normalization, same-device checks, deterministic multi-acquire with rollback on failure, all-path `has_lock()`, shared-inode heartbeat, whole-group stale breaking, and tolerant parsing of both manifest and legacy raw-claim lock contents. Update any helper functions that currently assume a link count of exactly two or that read the lock file as a plain claim-path string.

Run `uv run pytest tests/test_locking.py -q` after updating existing single-lock tests and adding new `lock_many(...)` coverage. Acceptance for this milestone is that the updated lock tests pass and a one-path `lock(...)` still behaves like the old API from a caller’s perspective while now using the new implementation internally.

### Milestone 4: Cross-process contention and documentation

At the end of this milestone, the full public behavior is proven under concurrent execution and the repository advertises the new API. Add a contention test in `tests/test_furu_locking_contention.py` where two processes request overlapping batches and the shared unresolved object is computed only once. Update `README.md` with a minimal usage example for top-level `load_or_create(...)` and a batched class hook. Re-run the full validation suite and update the living sections in this plan to match what actually shipped.

Run `uv run pytest -q`, `uv run ruff check`, and `uv run ty check`. Acceptance for this milestone is that the full test suite passes in the intended project environment, static checks pass, and the README example matches the final public API.

## Concrete Steps

Work from the repository root.

1. Read `src/furu/core.py`, `src/furu/logging.py`, `src/furu/locking.py`, `src/furu/validate.py`, `src/furu/__init__.py`, and the three main test files before editing so the implementation follows existing naming and error semantics.

2. Implement Milestone 1. Add the top-level overloads, the delegation method, and hook-mode validation. Add or update focused tests in `tests/test_core.py` first, then make them pass.

3. Implement Milestone 2. Refactor execution into helpers, then add logging fanout and same-thread self-reentry protection. Add or update tests for duplicates, mixed groups, existing-result skipping, and batch log scoping.

4. Implement Milestone 3. Introduce `lock_many(...)`, update `lock(...)`, switch the claim-file content to a manifest, and update stale-breaking helpers. Repair or rewrite tests that depend on the old plain-text lock format.

5. Implement Milestone 4. Add cross-process batch contention coverage, update `README.md`, and run the full validation commands.

The concrete commands to use are:

    uv run pytest tests/test_core.py -q
    uv run pytest tests/test_locking.py -q
    uv run pytest tests/test_furu_locking_contention.py -q
    uv run pytest -q
    uv run ruff check
    uv run ty check

When a new test is introduced before the supporting code exists, expect it to fail first. The milestone is complete only when the corresponding targeted test command passes.

## Validation and Acceptance

Acceptance is behavioral and must be observable through tests.

A successful implementation will satisfy all of the following checks. A batch-only class with only `_create_batched(...)` can be instantiated and can serve both `obj.load_or_create()` and `furu.load_or_create([obj])`. Passing `[a, b, a]` to the top-level function returns `[ra, rb, ra]` while computing only the unresolved unique objects. If `a` is already persisted before the call, `a` is skipped before locking and loaded from disk in the final output. If two concrete classes appear in one list, the public call succeeds and the engine partitions unresolved objects by concrete class behind the scenes. If one visible lock path in a multi-lock set is stolen or removed, `has_lock()` becomes false and persistence refuses to claim success. If two processes request overlapping unresolved batches, the shared object is computed once and both callers eventually observe the same stored result.

The final acceptance command set is:

    uv run pytest -q
    uv run ruff check
    uv run ty check

Do not declare success while only targeted tests pass. The feature is complete only when the full suite and the two static checks pass.

## Idempotence and Recovery

The edits in this plan are safe to apply incrementally. Re-running the targeted or full test commands is expected and should not corrupt repository state. The test suite already redirects Furu data into temporary directories through the pytest plugin in `src/furu/testing.py`, so repeated test runs should not pollute the working tree.

If a milestone fails halfway, keep the diff small and repair it in place rather than starting a parallel implementation. If the lock tests leave temporary files behind during local experimentation, delete only the temporary test directory, not repository files. If a lock-format change temporarily breaks many old tests at once, update the tests in the same commit as the corresponding implementation change so the working tree does not sit in an ambiguous mixed-format state.

## Artifacts and Notes

The intended public shape is:

    from typing import Sequence, overload

    class Furu[T]:
        def load_or_create(self, use_lock: bool = True) -> T:
            return load_or_create(self, use_lock=use_lock)

        def _create(self) -> T:
            raise NotImplementedError

        @classmethod
        def _create_batched(cls, objs: Sequence[Self]) -> list[T]:
            raise NotImplementedError

    @overload
    def load_or_create(obj: Furu[T], *, use_lock: bool = True) -> T: ...

    @overload
    def load_or_create(objs: Sequence[Furu[T]], *, use_lock: bool = True) -> list[T]: ...

A valid lock manifest must look like this conceptually:

    {
      "version": 2,
      "claim_path": "/abs/path/to/compute.lock.<host>.<pid>.<uuid>.claim",
      "lock_paths": [
        "/abs/path/to/artifact-a/.furu/compute.lock",
        "/abs/path/to/artifact-b/.furu/compute.lock"
      ]
    }

The executor helpers that should exist by the end of the work are conceptually:

    _normalize_batch_input(...)
    _dedupe_by_data_dir(...)
    _plan_execution(...)
    _group_pending_items(...)
    _write_running_metadata_for_pending_items(...)
    _store_result(...)
    _record_failure(...)
    _materialize_outputs(...)

Exact helper names may change, but the responsibilities must remain clearly separated.

## Interfaces and Dependencies

In `src/furu/core.py`, define a top-level overloaded `load_or_create(...)` and keep `Furu.load_or_create(...)` as a thin wrapper. Add a resolved-mode helper that can answer whether a concrete class is in single or batched mode. Add a lightweight same-thread reentry guard keyed by canonical `data_dir` strings so an object currently being computed cannot recursively ask for itself again while still unresolved.

In `src/furu/validate.py` or a nearby helper module, extend class validation to enforce the “at most one hook in the concrete class body” rule and the “inherit an existing mode or define one” rule. Do not hide this in runtime execution alone; class construction should fail early for obviously invalid classes.

In `src/furu/logging.py`, replace the single-path context with a tuple-of-paths context and add `_scoped_log_files(log_paths: tuple[Path, ...] | list[Path]) -> Iterator[None]`. `_ScopedFileHandler.emit(...)` must write each record to every active path in order. Nested scopes must override outer scopes cleanly.

In `src/furu/locking.py`, define:

    @contextmanager
    def lock_many(
        lock_paths: Iterable[Path],
        *,
        lifetime_s: float = DEFAULT_LIFETIME_S,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
        acquire_timeout_s: float | None = None,
        acquire_poll_interval_s: float | None = None,
    ) -> Iterator[Callable[[], bool]]:
        ...

Keep `lock(...)` public and implement it by delegating to `lock_many([lock_path], ...)`. Preserve the existing exception classes: `LockAcquireError`, `NotLockedError`, `LockLostError`, and `StaleLockRaceError`.

Update `src/furu/__init__.py` so `load_or_create` is part of `__all__`. Update `README.md` with a minimal example using the final public signature. No new third-party dependencies are needed for this work.

## Revision Note

Initial version created from the current repository state and the uploaded batching design report on 2026-04-03. The plan resolves several unstated implementation details up front, especially logging fanout, legacy stale-lock parsing, same-thread self-reentry handling, and the exact class-validation rule for `_create` versus `_create_batched`.
