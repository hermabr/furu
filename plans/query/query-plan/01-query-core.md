# plans/query/query-plan/01-query-core.md — Query AST + evaluator + type ops

## Scope
Implement a canonical JSON-serializable query AST + evaluator that can be used by:
- Python (directly)
- dashboard backend (scanner)
- dashboard API (POST body)

Files:
- `src/furu/query/ast.py`
- `src/furu/query/paths.py`
- `src/furu/query/types.py`
- `src/furu/query/eval.py`
- `src/furu/query/__init__.py`

## AST schema (exact v1)

### Node conventions
- Every node has `op: <string literal>`.
- The AST is a discriminated union on `op` (Pydantic v2).
- Model config:
  - `extra="forbid"`
  - `frozen=True`

### Scalar type (v1)
`Scalar = str | int | float | bool | None`

### Boolean nodes
- `{"op":"true"}`
- `{"op":"false"}`
- `{"op":"and","args":[Query,...]}` (min 1)
- `{"op":"or","args":[Query,...]}` (min 1)
- `{"op":"not","arg":Query}`

### Existence nodes
- `{"op":"exists","path":"config.seed"}`
- `{"op":"missing","path":"config.seed"}`

### Comparison nodes
- `{"op":"eq","path":..., "value": Scalar}`
- `{"op":"ne","path":..., "value": Scalar}`
- `{"op":"lt","path":..., "value": Scalar}`
- `{"op":"lte","path":..., "value": Scalar}`
- `{"op":"gt","path":..., "value": Scalar}`
- `{"op":"gte","path":..., "value": Scalar}`

### Range
- `{"op":"between","path":..., "low": Scalar, "high": Scalar, "inclusive": "both"|"left"|"right"|"none"}`

### Membership
- `{"op":"in","path":..., "values":[Scalar,...]}`
- `{"op":"nin","path":..., "values":[Scalar,...]}`

### String-ish
- `{"op":"contains","path":..., "value": Scalar, "case_sensitive": bool}`
- `{"op":"startswith","path":..., "prefix": str, "case_sensitive": bool}`
- `{"op":"endswith","path":..., "suffix": str, "case_sensitive": bool}`
- `{"op":"regex","path":..., "pattern": str, "flags": str}`

### Type-aware (uses serialized `__class__`)
- `{"op":"type_is","path":..., "type": "module.QualName"}`
- `{"op":"is_a","path":..., "type": "module.QualName"}`
- `{"op":"related_to","path":..., "type": "module.QualName"}`

Semantics:
- candidate is taken from the value at `path`:
  - if it’s a dict with key `__class__`, candidate type string = that value
  - else: false
- `type_is`: candidate string equals base string
- `is_a`: `issubclass(candidate_cls, base_cls)` (best-effort import)
- `related_to`: `issubclass(candidate_cls, base_cls) OR issubclass(base_cls, candidate_cls)`

## Path semantics (v1)
Implement a single resolver used everywhere:
- dot-separated segments
- dict traversal by string key
- list/tuple traversal by digit segment (0,1,2,...)
- missing anywhere returns a sentinel `PATH_MISSING`

Example:
- `config.dataset.name`
- `config.deps.0.__class__`

## Evaluator semantics (v1)
`matches(doc: dict, query: Query) -> bool`

Rules:
- comparisons return false if path is missing
- for `exists`: true iff path is not missing
- for `missing`: true iff path is missing
- for comparisons:
  - attempt numeric coercion when comparing `str` query values against numeric actuals:
    - `"0.001"` should compare equal to `0.001` when possible
  - for non-coercible types, fall back to strict Python comparisons only when valid
  - never raise due to a comparison type error; treat as false

Regex rules:
- only apply to string actual values
- compile with `re` flags parsed from `flags`

## Checklist
- [x] Add `src/furu/query/ast.py` with discriminated union nodes exactly as above
- [x] Add `src/furu/query/paths.py` implementing `get_path(doc, path) -> value|PATH_MISSING`
- [x] Add `src/furu/query/types.py`:
  - robust `resolve_type("a.b.C") -> type|None` (import longest module prefix, getattr chain)
  - handle enum-style strings like `"mod.Enum:VALUE"` by stripping `:VALUE` for type resolution
  - cache resolutions
- [x] Add `src/furu/query/eval.py` implementing `matches(doc, query)`
- [x] Add `src/furu/query/__init__.py` exporting:
  - `Query` type
  - node classes (optional)
  - `matches`
- [ ] Add small internal limits (API hardening):
  - [ ] max node count (e.g. 200)
  - [ ] max depth (e.g. 30)
  - implement as a helper `validate_query(query)` used by API route

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-02-14 | (start) |
| 2026-02-15 | Added `src/furu/query/ast.py` with frozen/forbid Pydantic node models and a discriminated `Query` union by `op`. |
| 2026-02-15 | Added `src/furu/query/paths.py` with `PATH_MISSING` sentinel and dot-path traversal across dict keys plus list indices. |
| 2026-02-15 | Added `src/furu/query/types.py` with cached `resolve_type(...)`, enum-style `:VALUE` stripping, longest module-prefix lookup, and getattr-chain type resolution. |
| 2026-02-15 | Added `src/furu/query/eval.py` implementing `matches(doc, query)` with boolean composition, path-aware operators, numeric string coercion, string predicates, regex, and type relationship ops over `__class__`. |
| 2026-02-15 | Added `src/furu/query/__init__.py` to export `Query`, query AST node classes, `Scalar`, and `matches` as the public query-core entrypoint. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-02-14 | — | initial |
