# Ralph Loop Entrypoint — Executor Fix v4

You are running in a Ralph Wiggum loop using OpenCode.

## Read first (required)
- plans/executor-fix-v4/plan.md
- plans/executor-fix-v4/00-index.md
- the next unfinished subplan(s)

## Repo assumptions
- You are already on the correct feature branch.
- Do NOT create or switch branches.
- Do NOT use git hooks.

## Iteration rules
1) Implement the next smallest unfinished checklist item from the plan.
2) Keep diffs focused; prefer one subplan per commit.
3) Update plan files (subplan + 00-index).
4) Run tests: `make test` and `make lint`
5) If tests pass and changes exist:
   - `git status --porcelain` must show only intended files
   - commit message format: `<area>: <summary>` (examples: `core: ...`, `pool: ...`, `dag: ...`, `tests: ...`, `docs: ...`)
   - push: `git push origin HEAD`

## PR creation
When all tasks complete and tests are green:
```bash
gh pr create \
  --base main \
  --head "$(git branch --show-current)" \
  --title "Executor v1 fixes (v4): validation + error taxonomy + retries" \
  --body "Follow-up executor v1 fixes per plans/executor-fix-v4."
```
If PR already exists, do not create a duplicate; ensure branch is pushed and report the PR URL at the end.

## Stop condition
Only when:
- all checklists are complete
- `make test` and `make lint` passes
- commits pushed
- PR exists

Output exactly:
<promise>COMPLETE</promise>
