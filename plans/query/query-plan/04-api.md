# plans/query/query-plan/04-api.md — Dashboard API: POST search endpoint

## Scope
Add a dashboard endpoint that accepts the AST query in JSON form.

Files:
- `src/furu/dashboard/api/routes.py`
- `src/furu/dashboard/api/models.py`

## Endpoint
Add:
- `POST /api/experiments/search`

Request body:
```json
{
  "query": { ... Query AST ... },
  "schema": "current" | "stale" | "any",
  "view": "resolved" | "original",
  "limit": 100,
  "offset": 0
}
```

Response:
- same shape as `GET /api/experiments` (ExperimentList)

## Validation
- Validate the query AST via Pydantic union parsing.
- Apply `validate_query(query)` limits (node count, depth) from 01.

## Backcompat
Keep `GET /api/experiments` unchanged.

## Checklist
- [x] Add request model `ExperimentSearchRequest` to `api/models.py`
- [x] Add `POST /api/experiments/search` route in `api/routes.py`
- [x] Wire it to call `scan_experiments(query=req.query, schema=req.schema, view=req.view, ...)`
- [x] Add API tests in `tests/dashboard/test_api.py` for POST endpoint:
  - [x] filter by result_status via AST
  - [x] filter by nested config field via AST
  - [x] type filter via AST

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-02-14 | (start) |
| 2026-02-15 | Added `ExperimentSearchRequest`, implemented `POST /api/experiments/search` with `validate_query(...)` limits + pagination passthrough, and covered AST result-status/nested-config/type filters in dashboard API tests. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-02-14 | — | initial |
