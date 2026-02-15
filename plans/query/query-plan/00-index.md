# plans/query/query-plan/00-index.md — v1 index, tracking, and implementation order

> This file is the **single tracker** for query/filtering v1 implementation.
> Each feature has a dedicated subplan with its own checklist + progress log.

## Quick navigation
- 01-query-core.md — AST schema + path resolver + evaluator + type ops
- 02-python-dsl.md — `Q` DSL that compiles to AST
- 03-scanner-integration.md — integrate query into `scan_experiments` + compile legacy filters
- 04-api.md — POST search endpoint + models
- 05-dashboard-frontend.md — minimal advanced filter support
- 06-tests-and-docs.md — tests + docs + examples

## Recommended implementation order
1. Query core (AST + eval) (01)
2. Python DSL (02)
3. Scanner integration (03)
4. API endpoint (04)
5. Frontend wiring (05)
6. Tests + docs (06)

## Global milestone checklist
- [x] M0: Add `src/furu/query/` package with AST + evaluator + type ops
- [x] M1: Add Python DSL `Q` and export it cleanly
- [ ] M2: Integrate query into `scan_experiments` while preserving old filters
- [ ] M3: Add `POST /api/experiments/search` (dashboard)
- [ ] M4: Add minimal dashboard UI support to submit an AST query
- [ ] M5: Add test coverage for query eval, DSL, scanner, API
- [ ] M6: Add docs/examples for Python + dashboard usage

## Progress Log (append-only)

| Date | Area | Summary |
|---|---|---|
| 2026-02-14 | — | (start) |
| 2026-02-15 | Completed query-core milestone tasks: path resolver + AST + evaluator/type helpers in `src/furu/query/` were added. | core scaffolding |
| 2026-02-15 | Completed query DSL milestone M1: added `src/furu/query/dsl.py`, exported `Q`, `TRUE`, and `FALSE` from `furu.query`, with AST composition support. | query usability |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-02-14 | — | initial |
