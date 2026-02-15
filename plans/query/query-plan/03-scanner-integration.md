# plans/query/query-plan/03-scanner-integration.md — Integrate Query into dashboard scanner

## Scope
Integrate the query engine into:
- `src/furu/dashboard/scanner.py::scan_experiments`

Must preserve:
- existing filter query params
- existing `config_filter="field=value"` behavior
- existing dashboard tests

## Design
### Add optional parameter
Add to `scan_experiments` signature:
- `query: furu.query.ast.Query | None = None`

### Build the query doc for evaluation
After `summary` and `metadata` are computed:
- doc = {
    "exp": summary.model_dump(mode="json"),
    "config": metadata.get("furu_obj"),
    "meta": metadata,
    "state": state.model_dump(mode="json"),
  }

Evaluate:
- if query is not None and not matches(doc, query): continue

### Backcompat: compile legacy filters into an AST
Instead of two filtering systems, compile old params into a Query AST and AND them with `query`.

Legacy mapping (recommended v1):
- `result_status` -> `eq("exp.result_status", ...)`
- `attempt_status` -> `eq("exp.attempt_status", ...)`
- `namespace_prefix` -> `startswith("exp.namespace", prefix)`
- `backend/hostname/user` -> `eq("exp.backend"/..., ...)`
- `migration_kind/policy` -> `eq("exp.migration_kind"/..., ...)`
- `schema`:
  - current => `ne("exp.is_stale", True)`
  - stale => `eq("exp.is_stale", True)`
  - any => no-op

Dates:
- v1 choice:
  - keep existing date filtering logic (started_after/before, updated_after/before) outside AST
  - OR compile to comparisons on `exp.started_at` / `exp.updated_at` once evaluator supports ISO datetime compare
Pick one; keep behavior identical.

Config filter:
- compile `config_filter="field.path=value"` to:
  - `eq("config.<field.path>", parsed_value)`
- parsing rules:
  - "true/false" -> bool
  - int -> int
  - float -> float
  - "null"/"none" -> None
  - else -> string

## Checklist
- [ ] Add `query` parameter to `scan_experiments`
- [ ] Build `doc` and evaluate `matches(doc, query)`
- [ ] Implement `compile_legacy_filters_to_query(...) -> Query | None`
- [ ] Ensure legacy filters remain equivalent (tests must stay green)
- [ ] Remove/stop using `_get_nested_value` for filtering (keep only if used elsewhere)

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-02-14 | (start) |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-02-14 | — | initial |
