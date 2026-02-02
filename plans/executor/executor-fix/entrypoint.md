# Ralph loop entrypoint — Executor v1.1 hardening

You are running in a Ralph Wiggum loop (OpenCode). This prompt will repeat each iteration.

## Read first (every iteration)
- `plans/executor-fix/plan.md`
- `plans/executor-plan-fix/00-index.md`
- Then open the next unfinished phase file in `plans/executor-plan-fix/`

These files are the source of truth.

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
4) Run: `make test` and `make lint`
5) Fix failures before proceeding.

### Git discipline (IMPORTANT)
6) If there are code/plan changes and tests pass:
   - ensure `git status --porcelain` shows only intended files
   - `git commit -am "<prefix>: <short message>"` (or stage explicitly)
   - `git push origin HEAD`

### Stop condition
Only when ALL plan checklists are complete and `make test` and `make lint` is green and changes are pushed, output exactly:

<promise>COMPLETE</promise>
