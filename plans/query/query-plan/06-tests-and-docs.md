# plans/query/query-plan/06-tests-and-docs.md — Tests + docs

## Scope
Add confidence that:
- query AST parses + evaluates correctly
- DSL produces correct AST
- scanner integration preserves old behavior
- API endpoint works
- docs show usage for Python + dashboard

## Tests to add

### Core query tests (run under `make test`)
Add new files under `tests/` (NOT under `tests/dashboard/` so they run in default suite):
- `tests/test_query_paths.py`
  - dict traversal
  - list index traversal
  - missing path behavior
- `tests/test_query_eval.py`
  - eq/ne/gt/gte/lt/lte
  - between inclusive modes
  - in/nin
  - exists/missing
  - contains/startswith/endswith
  - type_is/is_a/related_to using locally-defined classes
- `tests/test_query_dsl.py`
  - FieldRef path building: attrs + [0] + ["key"]
  - operators build correct nodes
  - `& | ~` composition and flattening
  - truthiness raises

### Dashboard integration tests (run under `make dashboard-test`)
- Update `tests/dashboard/test_scanner.py`:
  - add tests calling `scan_experiments(query=...)` using `furu.query.Q`
  - include nested config path filtering: `Q.config.dataset.name == "mnist"`
  - include type filtering using `dashboard.pipelines.Data/DataA/DataB`
- Update `tests/dashboard/test_api.py`:
  - POST `/api/experiments/search` tests (basic + nested + type)

## Docs
- Update `README.md` (or add a short doc file) with:
  - Python examples using `Q` + `scan_experiments(query=...)`
  - Dashboard advanced JSON example payloads
  - Notes on type filters relying on importability (until we add MRO indexing)

## Checklist
- [ ] Add core query tests under `tests/`
- [ ] Add dashboard scanner tests for `query=...`
- [ ] Add dashboard API tests for POST search
- [ ] Update README/docs with examples
- [ ] Ensure `make lint`, `make test`, `make dashboard-test` all green

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-02-14 | (start) |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-02-14 | — | initial |
