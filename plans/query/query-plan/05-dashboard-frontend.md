# plans/query/query-plan/05-dashboard-frontend.md — Minimal dashboard support for AST queries

## Scope
Add minimal UI support to send an AST query to the new endpoint.

Files:
- `dashboard-frontend/src/routes/experiments.tsx`

Notes:
- `dashboard-frontend/src/api/` is generated (gitignored).
- After backend route changes, run `make frontend-generate` before TypeScript changes.

## Minimal UI (v1)
Add an “Advanced” section to Experiments page:
- textarea (JSON) for the query AST
- Apply button that:
  - parses JSON
  - POSTs to `/api/experiments/search`
  - shows parse/server errors

Behavior:
- If advanced query is empty -> keep existing GET filtering behavior (no breaking change).
- If advanced query is present -> use POST endpoint and ignore legacy query params (or AND them—pick one).

## Checklist
- [x] Run `make frontend-generate` (ensures TS client types exist)
- [x] Add `advancedQueryJson` state in experiments page
- [x] Add textarea + Apply/Clear controls
- [x] Add POST call to `/api/experiments/search`:
  - prefer using generated hook once available
  - fallback: `fetch("/api/experiments/search", { method:"POST", body: JSON.stringify(...) })`
- [x] Keep pagination working (limit/offset)
- [x] Run `make frontend-lint` (and optionally `make frontend-test`)

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-02-14 | (start) |
| 2026-02-15 | Ran `make frontend-generate`, added advanced JSON AST query state + apply/clear UI to experiments page, wired POST `/api/experiments/search` via generated mutation hook with parse/server error handling, and preserved pagination for advanced mode. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-02-14 | — | initial |
