# plans/query/query-plan/02-python-dsl.md — Python DSL (`Q`) that compiles to AST

## Scope
Implement an ergonomic Python DSL that builds the AST from 01-query-core.

Files:
- `src/furu/query/dsl.py`
- `src/furu/query/__init__.py` (re-export)
- `src/furu/__init__.py` (optional re-export convenience)

## DSL surface (exact v1)

### Root
`Q` exposes namespaces:
- `Q.exp` -> FieldRef("exp")
- `Q.config` -> FieldRef("config")
- `Q.meta` -> FieldRef("meta")
- `Q.state` -> FieldRef("state")

### FieldRef path building
- attribute access: `Q.config.dataset.name` -> `"config.dataset.name"`
- index access:
  - `Q.config.deps[0]` -> `"config.deps.0"`
  - `Q.config["weird_key"]` -> `"config.weird_key"` (v1: forbid "." inside string keys)

### Operators => AST
- `==` -> `eq`
- `!=` -> `ne`
- `<` -> `lt`
- `<=` -> `lte`
- `>` -> `gt`
- `>=` -> `gte`

### Methods => AST
- `.exists()` -> `exists`
- `.missing()` -> `missing`
- `.between(low, high, inclusive="both")` -> `between`
- `.in_(...)` -> `in`
- `.not_in(...)` -> `nin`
- `.contains(value, case_sensitive=True)` -> `contains`
- `.startswith(prefix, case_sensitive=True)` -> `startswith`
- `.endswith(suffix, case_sensitive=True)` -> `endswith`
- `.regex(pattern, flags="")` -> `regex`

### Type methods
`T` can be a class, instance, or fully qualified string:
- `.type_is(T)` -> `type_is`
- `.is_a(T)` -> `is_a`
- `.related_to(T)` -> `related_to`

### Boolean composition
- `q1 & q2` -> `and` (flatten nested)
- `q1 | q2` -> `or` (flatten nested)
- `~q` -> `not`
- prevent accidental truthiness: `__bool__` should raise `TypeError`

## Examples (must work)
- `q = (Q.exp.result_status == "success") & Q.config.is_a("dashboard.pipelines.TrainModel")`
- `q = Q.config.lr.between(1e-4, 1e-3) & (Q.config.dataset.name == "mnist")`
- `q = Q.config.data.is_a(TextData)`
- `q = Q.config.data.related_to(DummyData)`

## Checklist
- [ ] Implement `FieldRef` + `Q` root in `src/furu/query/dsl.py`
- [ ] Ensure composition operators produce AST nodes from 01 (and flatten)
- [ ] Add `furu.query` exports:
  - [ ] `Q`
  - [ ] `TRUE` / `FALSE` convenience (optional)
- [ ] (Optional) Re-export `Q` from `furu/__init__.py` for convenience:
  - [ ] add to `__all__`

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-02-14 | (start) |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-02-14 | — | initial |
