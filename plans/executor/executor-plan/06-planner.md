# plans/executor-plan/06-planner.md — Dependency planner (DONE/IN_PROGRESS/TODO/FAILED)

## Scope

Implement a planner that:
- discovers dependencies (DAG) from roots
- classifies nodes into DONE / IN_PROGRESS / TODO / FAILED
- prunes traversal into DONE / IN_PROGRESS / FAILED nodes
- provides:
  - topo order for TODO nodes (for Slurm DAG submission)
  - ready set for TODO nodes (for local executor and pool controller)
  - a pending/completed summary view

## File: `src/furu/execution/plan.py`

### Data structures (v1)

```py
from dataclasses import dataclass
from typing import Dict, Set, Literal

Status = Literal["DONE", "IN_PROGRESS", "TODO", "FAILED"]

@dataclass
class PlanNode:
    obj: Furu
    status: Status
    spec_key: str
    deps_all: Set[str]
    deps_pending: Set[str]
    dependents: Set[str]

@dataclass
class DependencyPlan:
    roots: list[Furu]
    nodes: Dict[str, PlanNode]
```

### Classification function

Use existing state + cache semantics:
- DONE:
  - `obj.exists()` true and not always_rerun
- IN_PROGRESS:
  - state attempt status in {"queued","running"}
- FAILED:
  - state indicates failed attempt/result
- TODO:
  - otherwise

### Build plan (pruned traversal)

```py
def build_plan(roots: list[Furu]) -> DependencyPlan:
    nodes: dict[str, PlanNode] = {}
    stack = list(roots)
    seen = set()

    while stack:
        obj = stack.pop()
        h = obj._furu_hash
        if h in seen:
            continue
        seen.add(h)

        status = classify(obj)
        node = PlanNode(
            obj=obj,
            status=status,
            spec_key=obj._executor_spec_key(),
            deps_all=set(),
            deps_pending=set(),
            dependents=set(),
        )
        nodes[h] = node

        # prune traversal for DONE/IN_PROGRESS/FAILED
        if status != "TODO":
            continue

        deps = obj._get_dependencies(recursive=False)
        node.deps_all = {d._furu_hash for d in deps}
        for d in deps:
            stack.append(d)

    # compute deps_pending and dependents
    for h, node in nodes.items():
        if node.status != "TODO":
            continue
        node.deps_pending = {d for d in node.deps_all if d in nodes and nodes[d].status != "DONE"}

    for h, node in nodes.items():
        for d in node.deps_pending:
            nodes[d].dependents.add(h)

    return DependencyPlan(roots=roots, nodes=nodes)
```

### Topological order for TODO nodes (for Slurm DAG)

Use Kahn’s algorithm over TODO nodes (deterministic ordering):

```py
def topo_order_todo(plan: DependencyPlan) -> list[str]:
    todo = {h for h, n in plan.nodes.items() if n.status == "TODO"}
    indeg = {h: 0 for h in todo}

    for h in todo:
        node = plan.nodes[h]
        for d in node.deps_pending:
            if d in todo:
                indeg[h] += 1

    ready = sorted([h for h, deg in indeg.items() if deg == 0])
    out: list[str] = []

    while ready:
        h = ready.pop(0)
        out.append(h)
        for dep in plan.nodes[h].dependents:
            if dep not in todo:
                continue
            indeg[dep] -= 1
            if indeg[dep] == 0:
                ready.append(dep)
                ready.sort()

    if len(out) != len(todo):
        raise ValueError("Cycle detected in TODO dependency graph")
    return out
```

### Ready nodes helper (for local executor + pool)

A TODO node is ready when all deps are DONE (not merely IN_PROGRESS), because executor `.get()` is strict.

```py
def ready_todo(plan: DependencyPlan) -> list[str]:
    return sorted([
        h for h, n in plan.nodes.items()
        if n.status == "TODO" and all(plan.nodes[d].status == "DONE" for d in n.deps_pending)
    ])
```

## Checklist

- [x] Implement `build_plan` with pruning
- [x] Implement `topo_order_todo`
- [x] Implement `ready_todo`
- [x] Ensure classification uses repo’s real state representation

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Implement planner DAG builder, topo ordering, ready helper, and status classification. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
