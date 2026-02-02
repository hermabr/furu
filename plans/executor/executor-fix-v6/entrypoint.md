# plans/executor-fix-v6/entrypoint.md
# Ralph Loop Entrypoint — Furu Fix v6

You are running in a Ralph Wiggum loop using OpenCode.

## Read first (required)
- plans/executor-fix-v6/plan.md
- plans/executor-fix-v6/00-index.md
- the next unfinished subplan(s) referenced there

## Repo assumptions
- You are already on the correct feature branch for v6 follow-ups.
- Do NOT create or switch branches.
- Do NOT use git hooks.

## Iteration rules
1) Implement the next smallest unfinished checklist item from the plan.
2) Keep diffs focused; prefer one subplan per commit.
3) Update plan files (subplan + 00-index).
4) Run tests:
   - `make lint` and `make test`
   - plus any extra command specified in the subplan
5) If tests pass and changes exist:
   - ensure `git status --porcelain` shows only intended files
   - commit message format: `<area>: <summary>`
     - examples: `pool: ...`, `dag: ...`, `local: ...`, `core: ...`, `docs: ...`, `tests: ...`
   - push: `git push origin HEAD`

## PR creation
When all tasks complete and tests are green:
```bash
gh pr create \
  --base main \
  --head "$(git branch --show-current)" \
  --title "Furu fixes v6: retry semantics + pool heartbeat bounds + DAG job_id robustness" \
  --body "Follow-up fixes per plans/executor-fix-v6."
```
If PR already exists, do not create a duplicate; ensure branch is pushed and report PR URL in final output.

## Stop condition
Only when:
- all checklists are complete
- `make lint` and `make test` passes
- commits pushed
- PR exists

Output exactly:
<promise>COMPLETE</promise>
