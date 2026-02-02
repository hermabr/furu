# Ralph Loop Entrypoint — Executor Fix v2

You are running in a Ralph Wiggum loop using OpenCode.

## Read first (required)
- plans/executor-fix-v2/plan.md
- plans/executor-fix-v2/00-index.md
- The next unfinished subplan file(s) referenced there

## Repository assumptions
- You are already on the correct feature branch.
- Do NOT create or switch branches.
- Do NOT use git hooks.

## Rules for each iteration
1) Implement the next smallest unfinished checklist item from the plan subfiles.
2) Keep diffs small and focused.
3) Update the relevant plan subfile(s) and the master checklist in `00-index.md`.
4) Run tests:
   - `make test` and `make lint`
5) If tests pass and there are changes:
   - `git status --porcelain` must show only intended files
   - Commit with a concise message prefixed by the subsystem (examples: `pool: ...`, `dag: ...`, `plan: ...`, `core: ...`, `tests: ...`)
   - Push: `git push origin HEAD`

## Pull request creation
When ALL tasks in this plan are complete and `make test` and `make lint` is green:
- Create or update a PR via GitHub CLI:
  ```bash
  gh pr create \
    --base main \
    --head "$(git branch --show-current)" \
    --title "Executor v1 fixes (v2): pool/dag correctness + retry semantics" \
    --body "Fixes executor v1 blocking issues per plans/executor-fix-v2."
  ```
- If the PR already exists, do NOT create a duplicate; instead ensure branch is pushed and report PR URL in the final output.

## Stop condition
Only when:
- all checklists are complete
- tests pass
- commits are pushed
- PR exists

Output exactly:
<promise>COMPLETE</promise>
