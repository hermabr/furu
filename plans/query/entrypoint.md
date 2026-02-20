# Ralph loop entrypoint — Experiment Query/Filtering v1 (AST + Python DSL)

You are running in a Ralph Wiggum loop (OpenCode). This prompt will repeat each iteration.

## Read first (every iteration)
- `plans/query/plan.md`
- `plans/query/query-plan/00-index.md`
- Then open the next unfinished file in `plans/query/query-plan/` (in order)

These documents are the source of truth.

## Repo assumptions
- You are already on the correct feature branch.
- Do NOT create/switch branches.
- No git hooks.
- You must commit and push yourself.

## Loop rules (follow strictly)

### Implementation
1) Pick the **next unchecked** item from the plan files.
2) Implement the **smallest coherent chunk**.
3) Update the plan file(s):
   - mark completed checkboxes
   - add a short progress note with what changed

### Verification
4) Run (at minimum):
   - `make lint`
   - `make test`
   - `make dashboard-test`
5) Fix failures before proceeding.

### Git discipline (IMPORTANT)
6) If there are code/plan changes and tests pass:
   - ensure `git status --porcelain` shows only intended files
   - commit with a concise message prefix:
     - `query: ...` for core query engine
     - `dashboard: ...` for scanner/api
     - `frontend: ...` for dashboard UI
     - `tests: ...` for test-only changes
   - `git push origin HEAD`

### Stop condition
Only when ALL plan checklists are complete and `make lint`, `make test`, and `make dashboard-test` are green and changes are pushed, output exactly:

<promise>COMPLETE</promise>
