# plans/executor-plan/08-slurm-dag.md — Slurm per-node DAG submission (`afterok`)

## Scope

Implement per-node Slurm submission:
- one Slurm job per TODO node
- dependencies wired via `afterok:<jobid1>:<jobid2>...`
- submitit logs under `<submitit_root>/nodes/<spec_key>/...`
- uses `specs` mapping (must include `"default"`)

## Files

- `src/furu/execution/submitit_factory.py` — create submitit executor from `SlurmSpec`
- `src/furu/execution/slurm_dag.py` — submit DAG using topo order + afterok

## `submitit_factory.py` (v1)

```py
from pathlib import Path
from furu.execution.paths import submitit_logs_dir
from furu.execution.slurm_spec import SlurmSpec

def make_executor_for_spec(spec_key: str, spec: SlurmSpec, *, kind: str, submitit_root: Path | None):
    import submitit
    folder = submitit_logs_dir(kind, spec_key, override=submitit_root)
    folder.mkdir(parents=True, exist_ok=True)

    ex = submitit.AutoExecutor(folder=str(folder))
    params = dict(
        timeout_min=spec.time_min,
        slurm_partition=spec.partition,
        cpus_per_task=spec.cpus,
        mem_gb=spec.mem_gb,
    )
    if spec.gpus:
        params["gpus_per_node"] = spec.gpus
    if spec.extra:
        params.update(spec.extra)

    ex.update_parameters(**{k: v for k, v in params.items() if v is not None})
    return ex
```

## `slurm_dag.py` submission algorithm (v1)

Public API:

```py
@dataclass
class SlurmDagSubmission:
    plan: DependencyPlan
    job_id_by_hash: dict[str, str]
    root_job_ids: dict[str, str]

def submit_slurm_dag(
    roots: list[Furu],
    *,
    specs: dict[str, SlurmSpec],
    submitit_root: Path | None = None,
) -> SlurmDagSubmission:
    ...
```

Steps:

1. Validate `specs` contains `"default"`.
2. `plan = build_plan(roots)`
3. If any FAILED node is required → raise (do not submit).
4. `order = topo_order_todo(plan)`
5. For each TODO node in order:
   - Resolve dependency job ids:
     - DONE deps: ignore
     - TODO deps: use `job_id_by_hash`
     - IN_PROGRESS deps: read `job_id` from state or job pickle (poll briefly)
   - Create executor for node’s spec key:
     - `spec_key = plan.nodes[h].spec_key`
     - `spec = specs[spec_key]` (raise KeyError with helpful message if missing)
   - If deps exist:
     - set `slurm_additional_parameters={"dependency": "afterok:" + ":".join(dep_job_ids)}`
   - Submit:
     - `adapter = SubmititAdapter(executor)`
     - `job = obj._submit_once(adapter, directory=obj._base_furu_dir(), on_job_id=None)`
   - Wait until job id becomes available in state (important for dependents).
   - Store mapping.

Notes:
- Avoid dependency-parameter leakage between submissions: easiest is a fresh executor per node submission.
- The Submitit watcher race fix (05) is required for reliable job_id recording.

## Checklist

- [x] Implement submitit executor factory from SlurmSpec
- [x] Implement `submit_slurm_dag` with topo order and afterok wiring
- [x] Ensure `specs["default"]` required and missing spec keys raise clearly
- [x] Implement robust job_id retrieval (poll state; fallback to job pickle)

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Implement submitit executor factory, Slurm DAG submission, and tests. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
