# 05 — Local executor retry parity vs pool retry semantics

## Feedback / question
Local executor fails fast on compute errors; pool executor does bounded retries for compute failures.
Decide whether parity is intended or document the difference clearly.

## Options
A) Add bounded local retries (recommended if parity desired):
   - On compute failure of a node, if retry_failed=True, retry the node up to `max_compute_retries` (same config as pool).
   - Be careful to avoid retrying protocol failures (e.g., missing deps due to plan bug).

B) Keep local fail-fast but document explicitly:
   - local executor: stops at first compute failure
   - pool executor: bounded retries for compute failures
   - rationale: local is for fast debugging; pool is for long runs

## Implementation steps
- [x] Choose A or B.
- [x] Implement retry loop around node execution in `run_local` for compute failures only.
- [x] Add tests for local retry (fail once then succeed).

## Acceptance criteria
- Behavior is explicit and tested.
- No surprising divergence between local and pool without docs.

## Progress log
- Status: [x] done
- Notes:
  - 2026-01-23: added local compute retry loop bounded by max_compute_retries and tests.
