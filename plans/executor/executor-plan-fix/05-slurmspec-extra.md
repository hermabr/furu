# 05 — SlurmSpec.extra: support nested submitit configs

## Why

`SlurmSpec.extra` typing is currently too narrow (scalar-only), but submitit supports nested dict values
(e.g., `slurm_additional_parameters` is itself a dict). This forces type-ignore workarounds for valid configs.

Relevant files:
- `src/furu/execution/slurm_spec.py:7`
- `src/furu/execution/submitit_factory.py:28`

## Tasks

- [x] Widen type of `SlurmSpec.extra` to allow nested values:
  - e.g. `dict[str, Any] | None`

- [x] Ensure `submitit_factory` merges `extra` into `executor.update_parameters(...)` safely.

- [x] Add a test (unit-level):
  - create a SlurmSpec with `extra={"slurm_additional_parameters": {"qos": "foo"}}`
  - assert executor factory passes nested dict to update_parameters (use a fake submitit executor)

## Acceptance criteria

- Users can set nested submitit parameters through SlurmSpec.extra without type ignores.

## Progress log

| Date | Summary |
|---|---|
| 2026-01-22 | Allow nested extra values in SlurmSpec, pass through submitit update_parameters, and add nested-extra test coverage. |

## Plan changes

| Date | Change | Why |
|---|---|---|
|  |  |  |
