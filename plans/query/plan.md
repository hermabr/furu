# Experiment Query/Filtering v1 — AST + Python DSL

## Goal
Add a robust, composable filtering system so we can easily:
- find experiments by status/namespace/backend/etc
- filter on deep config fields (dot paths + list indices)
- filter by ranges (`between`, `gt/lte`, etc)
- filter by type relationships using serialized `__class__` markers:
  - exact type (`type_is`)
  - include subclasses (`is_a`)
  - include ancestors OR descendants (`related_to`)

This must be:
- **easy in Python** via a small DSL (`Q.config.lr.between(...) & Q.exp.result_status == "success"`)
- **easy for the dashboard** by sending a JSON AST to a new API endpoint.

## Non-goals (v1)
- full text search / fuzzy search
- DB indexing / query compilation to SQL (filesystem scan stays the source of truth)
- joins/aggregations across runs (e.g. group-by)
- permissions / multi-user auth

## Public surfaces (v1)
### Python
- `furu.query` package:
  - AST models (JSON-serializable, Pydantic v2, discriminated union)
  - evaluator `matches(doc, query)`
  - Python DSL `Q` that builds AST nodes
- Dashboard scanner:
  - `scan_experiments(..., query: Query | None = None)` (optional)
  - existing query params remain supported (backcompat)

### Dashboard API
- Add `POST /api/experiments/search` accepting `{ query, schema, view, limit, offset }`
- Keep `GET /api/experiments` unchanged (backcompat and simple filters)

### Dashboard frontend
- Add minimal “Advanced filter (JSON)” support so the dashboard can send the AST.
  (Full visual filter builder can be a later iteration.)

## Query document model (namespaces)
Each experiment gets evaluated against a JSON-ish doc with:
- `exp.*`: `ExperimentSummary.model_dump(mode="json")` (includes alias override fields)
- `config.*`: `metadata["furu_obj"]` (serialized config dict, includes `__class__`)
- `meta.*`: raw metadata dict (optional for future)
- `state.*`: `_FuruState.model_dump(mode="json")` (optional for future)

Paths are dot-separated; digit segments index lists (e.g. `config.deps.0.name`).

## Compatibility promise
- All existing dashboard tests (scanner + API) must remain green.
- Existing filters must behave the same:
  - `result_status`, `attempt_status`, `namespace_prefix`, etc.
  - `config_filter="field=value"` must keep working.
