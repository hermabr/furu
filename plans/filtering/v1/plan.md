Below is a detailed, codebase-aware plan for implementing an **AST-based filtering system with a Python DSL**, that also becomes **easy to support in the dashboard**.

I’m going to anchor everything in your existing structure:

* Backend filtering today happens in **`src/furu/dashboard/scanner.py::scan_experiments()`** (simple field filters + `config_filter="a.b=value"`).
* The dashboard calls **`GET /api/experiments`** with query params (see **`src/furu/dashboard/api/routes.py`**).
* Frontend filters are hardcoded in **`dashboard-frontend/src/routes/experiments.tsx`**.

The plan is to introduce a **single canonical filter representation (AST)** that:

1. can be built ergonomically in Python (DSL), and
2. can be sent from the dashboard as structured JSON (no parsing, easy UI builder), and
3. can be evaluated efficiently during scans (and later compiled to an index/DB if you want).

---

## 0) What you already have (and what we’ll reuse)

### Current “filtering” surface area

* **Dashboard backend**: `scan_experiments(...)` supports:

  * `result_status`, `attempt_status`, `namespace_prefix`, `backend`, `hostname`, `user`,
  * `started_after`, `started_before`, `updated_after`, `updated_before`,
  * and one special `config_filter="field.path=value"`, implemented by:

    * `_get_nested_value(furu_obj, config_field)` (dict-only traversal)
    * `str(actual_value) == config_value`

### Existing “dotted path semantics”

You already have a solid dotted path resolver pattern in:

* **`src/furu/testing.py::override_results_for()`**
  It supports:

  * numeric segments for list indices (`deps.0`)
  * key segments for mappings (`deps.key`)

This is perfect to reuse for query paths (just applied to JSON-ish dict/list structures).

### Where type information currently lives

* Serialized config is stored in metadata: `metadata["furu_obj"]`
* FuruSerializer adds `"__class__": "module.QualName"` for each serialized chz object.
* Dashboard DAG already reads dependencies using `"__class__"` markers.
* Dashboard **does not** currently compute inheritance (there’s a stub `_get_class_hierarchy()`).

So the substrate for “type filters” already exists: `"__class__"` strings.

---

## 1) Introduce a canonical Query AST (core module)

### Create a new package

Add:

```
src/furu/query/
  __init__.py
  ast.py
  dsl.py
  eval.py
  paths.py
  types.py
  schema.py   (optional, for dashboard field discovery)
```

This is not “dashboard-only”; it’s the shared core representation.

### AST design goals

* JSON-serializable (dashboard)
* Safe (no eval)
* Supports:

  * `and/or/not`
  * equality / inequality
  * comparisons / ranges
  * existence
  * deep paths
  * type relationships (subclass, related lineage)

### Suggested AST node set (minimal but expressive)

**Logical**

* `{"op": "and", "args": [Query, ...]}`
* `{"op": "or", "args": [Query, ...]}`
* `{"op": "not", "arg": Query}`

**Scalar comparisons**

* `eq`, `ne`, `lt`, `lte`, `gt`, `gte`
* `between` (inclusive by default): `{"op": "between", "path": "...", "low": X, "high": Y}`

**Membership / string**

* `in`: `{"op": "in", "path": "...", "values": [..]}`
* `contains`: string substring, list membership (explicit), set membership, etc.
* `startswith`, `endswith`
* `regex` (optional; if you add it, implement with `re` carefully + timeouts/limits)

**Null/missing**

* `exists`: `{"op": "exists", "path": "..."}`
* `is_null`: `{"op": "is_null", "path": "..."}`
* (or combine via exists/not exists)

**Type relationship operators**
These are key for your subclass request:

* `is_a` (candidate is instance/subclass of base)

  * `{"op": "is_a", "path": "config.data", "type": "my_pkg.TextData"}`
* `related_to` (candidate shares direct lineage with base: either ancestor or descendant)

  * `{"op": "related_to", "path": "config.data", "type": "my_pkg.DummyData"}`

Semantics:

* `is_a`: True if `issubclass(candidate_cls, base_cls)`
* `related_to`: True if `issubclass(candidate_cls, base_cls) OR issubclass(base_cls, candidate_cls)`

That matches your “filter from DummyData but also get TextData” example.

### Where these paths point (your “document model”)

Define a “query document” per experiment (not exposed, internal to evaluation):

* `exp.*` → fields from `ExperimentSummary` (namespace, result_status, attempt_status, backend, user, timestamps, is_stale, migration_kind, …)
* `config.*` → `metadata["furu_obj"]` (your serialized config)
* `meta.*` → top-level metadata fields (git_commit, timestamp, etc)
* `state.*` → state.json (result, attempt, etc) if you want deeper attempt filters later

This is important because `config.name` and `exp.namespace` are distinct.

---

## 2) Implement dotted-path access once (reusing your existing semantics)

### Add `src/furu/query/paths.py`

Reimplement (or directly adapt) the traversal rules from `src/furu/testing.py` but for JSON-ish structures:

* segment split by `.`
* dict:

  * prefer string key match
  * else if segment is digit, try int key (rare for your serialized objects, but consistent)
* list/tuple:

  * segment must be digit, index into sequence
* missing segment returns a sentinel `MISSING`

This replaces `scanner._get_nested_value()` which currently:

* only traverses dicts
* only returns primitives

Your new path-getter should return any JSON value, and the evaluator decides what ops are allowed.

---

## 3) Type resolution & subclass checks

### Add `src/furu/query/types.py`

You need a robust resolver for class strings like `"dashboard.pipelines.DataA"`.

Don’t use `rpartition(".")` like `FuruSerializer.from_dict` does (it can break for nested qualnames). Instead:

1. Split into module path candidates and attribute chain.
2. Try import progressively:

   * import the *module portion* (longest prefix that imports),
   * then getattr the remaining segments as nested attributes.

Cache results:

* `class_str -> type | None`
* `class_str -> mro list[str]` (optional but useful)

### How to evaluate `is_a` & `related_to`

For a value at `path`:

* if it’s a dict with `__class__`, use that string as candidate type
* if it’s a string and looks like a class path, optionally treat it as candidate type
* else false

Then:

* `is_a(candidate, base)`
* `related_to(candidate, base)` using symmetric check above

This directly supports your Data/DataA/DataB pattern from **`tests/dashboard/pipelines.py`**.

---

## 4) Python DSL that builds the AST (the part you want for ergonomics)

### Add `src/furu/query/dsl.py`

Provide a “field reference” object that accumulates a path:

```python
Q.exp.result_status
Q.config.dataset.name
Q.config.data  # points to a nested config object dict
```

Then operator overloads build AST nodes:

* `Q.exp.result_status == "success"` → `Eq(path="exp.result_status", value="success")`
* `Q.config.lr.between(1e-4, 1e-3)` → `Between(path="config.lr", low=..., high=...)`
* `Q.config.data.is_a(TextData)` → `IsA(path="config.data", type="my_pkg.TextData")`
* `Q.config.data.related_to(DummyData)` → `RelatedTo(path="config.data", type="...DummyData")`

Also support:

* `&` and `|` to build `and` / `or`
* `~expr` to build `not`

### DSL example aligned with your use cases

#### “all TrainModel experiments that are successful”

```python
from furu.query import Q

q = Q.config.is_a(TrainModel) & (Q.exp.result_status == "success")
```

#### “lr in range, and dataset name is mnist”

```python
q = Q.config.lr.between(1e-4, 1e-3) & (Q.config.dataset.name == "mnist")
```

#### “filter for TextData and all subclasses”

```python
q = Q.config.data.is_a(TextData)
```

#### “from DummyData, get everything related including TextData”

```python
q = Q.config.data.related_to(DummyData)
```

### Export surface

Update `src/furu/__init__.py` to export:

* `Q`
* `Query` / `Expr` types
* maybe `parse_query(...)` later, but AST means you don’t *need* a parser

---

## 5) Evaluator that applies a query AST to an experiment doc

### Add `src/furu/query/eval.py`

Core function:

* `matches(doc: dict, query: Query) -> bool`

Implementation:

* recursively evaluate `and/or/not`
* for leaf ops:

  * get value via `paths.get(doc, path)`
  * handle missing and null properly
  * apply comparisons with light coercion:

    * numeric comparisons: if `actual` is number and `value` is string numeric, coerce
    * datetime comparisons: if strings parse as ISO datetimes, compare as datetimes (optional but nice)

Limits / safety:

* enforce max AST depth / max node count in model validation (prevents pathological dashboard input)

---

## 6) Wire the AST into your existing scanner (minimal disruption, big gain)

### Modify `src/furu/dashboard/scanner.py`

#### Step 6.1: Add an optional `query` parameter

Something like:

```python
def scan_experiments(..., query: Query | None = None, ...) -> list[ExperimentSummary]:
```

#### Step 6.2: Build a doc object per experiment

Right now `scan_experiments()` already has:

* `summary` (ExperimentSummary)
* `metadata` raw dict
* `state` internal model `_FuruState`

Create:

```python
doc = {
  "exp": summary.model_dump(mode="json"),
  "config": metadata.get("furu_obj"),
  "meta": metadata,
  "state": state.model_dump(mode="json"),
}
```

Then:

* if `query` is present: `if not matches(doc, query): continue`

#### Step 6.3: Backward compatibility by compiling old query params into AST

Instead of maintaining two filter systems, convert the old parameters into a query AST and `AND` them together.

Example:

* `result_status=success` → `eq(exp.result_status, "success")`
* `namespace_prefix=x` → `startswith(exp.namespace, x)` (or a dedicated `prefix` op)
* old `config_filter="lr=0.001"` → `eq(config.lr, 0.001)` (attempt parse float/int/bool)

This removes duplicated filter logic and lets you gradually migrate frontend calls.

**This is the single highest-leverage refactor in scanner.py**.

#### Step 6.4: Replace `_get_nested_value`

Once AST is in place, you no longer need the scanner’s `_get_nested_value()` (or keep it for now, but long-term delete).

---

## 7) Add API support in a way that’s dashboard-friendly

### Update `src/furu/dashboard/api/models.py`

Add models:

* `Query` AST models (ideally imported from `furu.query.ast`)
* `ExperimentSearchRequest`:

  * `query: Query | None`
  * `view: str = "resolved"`
  * `schema: Literal["current","stale","any"] = "current"`
  * `limit`, `offset`
  * maybe `sort: list[SortSpec]` later

### Update `src/furu/dashboard/api/routes.py`

#### Add a new endpoint (recommended)

Keep existing `GET /api/experiments` for now, but add:

* `POST /api/experiments/search`

Request body is JSON, which is perfect for AST.

Inside:

* compile any legacy query params if you still accept them (optional)
* call `scan_experiments(query=...)`

This avoids URL-length issues and is easiest for the React dashboard.

#### Optionally: add `filter` param to the GET endpoint

For shareable URLs:

* `GET /api/experiments?filter=<base64url(json)>`
* decode & validate to Query AST

You can keep simple filters in the URL but allow advanced filters too.

---

## 8) Dashboard frontend plan (incremental, practical)

Right now filters are “form inputs → query params”.

With AST support you can evolve in stages:

### Stage A: Keep the current filter UI, but send AST

In `dashboard-frontend/src/routes/experiments.tsx`:

* Build AST in the frontend from the filter controls (AND everything)
* Call `POST /api/experiments/search` instead of GET

This keeps UI identical but moves the backend to the new filter engine.

### Stage B: Add an “Advanced filters” builder UI

Add a component like:

* `FilterBuilder`

  * Supports groups (AND/OR)
  * Rows: (field selector, operator selector, value editor)
  * Produces AST directly

Because you picked AST, the UI doesn’t need to parse text.
It just *creates structured nodes*.

### Stage C: Type-aware operators in UI

For rows where operator is `is_a` or `related_to`:

* Show:

  * a type picker (autocomplete) populated by an API endpoint that returns known classes
  * OR allow manual entry of fully-qualified type string

**Add backend endpoint**:

* `GET /api/types?scope=config` returns distinct `__class__` strings found in `metadata["furu_obj"]` across stored experiments.

This is simple to implement: scan once, cache, return list.

---

## 9) Optional but strongly recommended: store type lineage in metadata for “offline dashboard”

If you ever run the dashboard on a machine that can’t import the original code (common), `issubclass()` checks will fail.

You already handle a similar issue for schema staleness:

* `_current_schema_key()` returns None if the module can’t be imported.

For type filters, you have 3 options:

### Option 1 (fastest): “best effort imports” only

* Type filters only work if code is importable.
* If not importable, `is_a`/`related_to` evaluates to false (or “unknown”).

### Option 2 (recommended): Add a `type_index` to metadata at creation time

Update `src/furu/storage/metadata.py`:

* Extend `FuruMetadata` with an optional field like:

```python
type_index: dict[str, dict] | None
```

Where keys are config paths (`""`, `"dataset"`, `"data"`, `"deps.0"`, etc) and values include:

* `class`: `"my_pkg.DataA"`
* `mro`: `["my_pkg.DataA", "my_pkg.Data", "furu.core.furu.Furu", "object"]`

Then:

* `is_a(path, "my_pkg.Data")` becomes `"my_pkg.Data" in doc["meta"]["type_index"][path]["mro"]`
* No imports required at query time.

This is very dashboard-friendly.

Important note: because `FuruMetadata` currently has `extra="forbid"`, you must:

* add the field to the model
* ensure older metadata files (without it) still validate (default None)

### Option 3: build a separate index file next to the dashboard cache

This is “later”; Option 2 is simpler.

---

## 10) Update DAG inheritance (bonus: aligns with your TODO)

You already left a stub:

* `src/furu/dashboard/scanner.py::_get_class_hierarchy()`

Once you have `types.py` resolver and/or `type_index`, you can implement this properly:

* For each node’s `full_class_name`, compute its base class within your project (or within `furu.Furu`) and return `parent_class`.
* Add “inheritance edges” in DAG (separate edge type or reuse with a `kind`).

This immediately supports “show subclass relationships” from your TODO and makes the “related experiments” concept visible in UI.

---

## 11) Tests you should add (you already have perfect fixtures)

### Query engine unit tests

Create `tests/query/test_eval.py` covering:

* eq/lt/between/in/exists
* deep path traversal including list indices
* type ops using your existing inheritance fixtures:

  * `Data`, `DataA`, `DataB` in `tests/dashboard/pipelines.py`

### Scanner tests

In `tests/dashboard/test_scanner.py` add cases like:

* Create `DataA(...)`, `DataB(...)`, `Train(data=...)` experiments
* Query assertions:

1. “is_a(Data)” returns both DataA and DataB experiments
2. “related_to(DataA)” returns DataA and Data (if present) and subclasses (if any)

### API tests

In `tests/dashboard/test_api.py`:

* Add `POST /api/experiments/search` tests
* Verify it matches the old behavior for existing simple filters
* Verify a nested config query works (e.g. `config.lr between`)

---

## 12) Practical migration strategy (to keep you moving)

If you want the most “straight line” rollout:

1. **Implement AST + evaluator + scanner integration**
   (no frontend changes yet)
2. Update `GET /api/experiments` to accept a `filter` param (optional) and/or add `POST /api/experiments/search`
3. Update frontend to call the new endpoint, but keep the same UI (compile UI state → AST)
4. Add type ops (`is_a`, `related_to`) and a minimal UI affordance (even a text input for class path at first)
5. Add proper filter builder UI later

---

## What I would do first in your repo (concrete checklist)

### Backend core

* [ ] Create `src/furu/query/ast.py` (pydantic models, discriminated union)
* [ ] Create `src/furu/query/paths.py` (adapt traversal semantics from `src/furu/testing.py`)
* [ ] Create `src/furu/query/types.py` (robust import + caching)
* [ ] Create `src/furu/query/eval.py` (matches(doc, query))
* [ ] Create `src/furu/query/dsl.py` (Q + operator overloads producing AST)
* [ ] Export `Q` (and AST types) from `src/furu/__init__.py`

### Dashboard backend integration

* [ ] Update `scan_experiments()` in `src/furu/dashboard/scanner.py` to accept `query`
* [ ] Compile existing query params into AST (AND them) so old behavior remains
* [ ] Add `POST /api/experiments/search` in `src/furu/dashboard/api/routes.py`
* [ ] Add request model(s) in `src/furu/dashboard/api/models.py`

### Dashboard frontend integration (minimum)

* [ ] Swap to `POST /api/experiments/search` and send AST, keep UI unchanged
* [ ] Add “Advanced filter” JSON textarea (quick win) OR small builder

### Type support (optional, but high value)

* [ ] Add `GET /api/types` endpoint returning distinct config `__class__` strings
* [ ] Add metadata `type_index` later if you want offline subclass filters
