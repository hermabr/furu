# Executor v1.1 hardening plan

This `plans/` bundle is designed for a **Ralph Wiggum loop**: an agent repeatedly reads these files, implements the next unchecked item, runs tests, updates the checkboxes/logs, commits, pushes, and repeats until done.

## What this plan fixes

This plan addresses the following problems in the current executor implementation:

- Slurm pool can **spin forever** when a task lands in `queue/failed` without updating Furu state (e.g., spec mismatch / missing payload). The controller never inspects failed queue entries.
- Executor schedulers currently **ignore `FURU_CONFIG.retry_failed`**, so failed nodes can’t be retried under executors even when retry is enabled.
- Slurm DAG `afterok` wiring assumes **submitit backend** for IN_PROGRESS nodes; non-submitit IN_PROGRESS leads to confusing timeouts.
- `_wait_for_job_id()` can still be brittle around queued→running attempt switches.
- Slurm pool has **no requeue/recovery** for tasks stranded in `queue/running` after worker crashes.
- `SlurmSpec.extra` typing is too narrow for nested submitit configs.
- Some executor semantics should be explicitly documented (breaking changes, always_rerun behavior), and planning loops should avoid noisy/expensive `exists()` checks.

## Navigation

- Start here: **`plans/executor-plan-fix/00-index.md`**
- Run loop: **`plans/executor-fix/entrypoint.md`**
- Each phase has its own file under `plans/executor-plan-fix/`.

## Progress tracking rules

- Mark completed items as `[x]`.
- If partially done, mark as `[~]` and add a note.
- Add a short entry to the **Progress log** in the relevant phase file.
- If you change the plan, record it in **Plan changes** in the relevant file.

## Definition of done

All checklist items across `plans/executor-plan-fix/*.md` are complete, tests pass (`make test` and `make lint`), changes are committed and pushed, and the agent outputs:

`<promise>COMPLETE</promise>`
