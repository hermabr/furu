# Ralph Loop Entrypoint — Executor Fix v3

You are running in a Ralph Wiggum loop using OpenCode.

## Read first (required)
- plans/executor-fix-v3/plan.md
- plans/executor-fix-v3/00-index.md
- The next unfinished subplan file(s) referenced there

## Repo assumptions
- You are already on the correct feature branch.
- Do NOT create or switch branches.
- Do NOT use git hooks.

## Iteration rules
1) Implement the next smallest unfinished checklist item from the plan subfiles.
2) Keep diffs small and focused.
3) Update plan files:
   - the subplan(s) you touched
   - `00-index.md`
4) Run tests:
   - `make test` and `make lint`
5) If tests pass and there are changes:
   - ensure `git status --porcelain` shows only intended files
   - commit with a concise message prefixed by subsystem (examples: `pool: ...`, `dag: ...`, `plan: ...`, `core: ...`, `tests: ...`)
   - push: `git push origin HEAD`

## Pull request
When all tasks are done and tests are green:
- Create or update a PR using GitHub CLI:
  ```bash
  gh pr create \
    --base main \
    --head "$(git branch --show-current)" \
    --title "Executor v1 fixes (v3): pool retries + in-progress robustness" \
    --body "Fixes executor v1 blocking issues per plans/executor-fix-v3."
  ```
- If the PR already exists, do NOT create a duplicate; ensure branch is pushed and report PR URL in final output.

## Stop condition
Only when:
- all checklists are complete
- tests pass
- commits are pushed
- PR exists

Output exactly:
<promise>COMPLETE</promise>
