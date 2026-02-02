# 08 — Changelog updates

## Issue
Multiple user-visible changes are not reflected in CHANGELOG.md:
- dashboard addition/changes
- state/metadata refactors and compatibility notes
- executor v1 behavior changes (get() replacement, strictness)
- public furu_hash property reintroduction

## Desired behavior
- Add entries describing:
  - what changed
  - breaking changes and migrations
  - any compatibility notes (metadata extra forbid, etc.)
- Ensure version section matches release plan.

## Implementation steps
- [x] Update CHANGELOG.md with a new unreleased section (or appropriate version section).
- [x] Mention: furu_hash property, dashboard assets packaging, lock bug fix, executor pool heartbeat policy.

## Progress log
- Status: [x] done
- Notes: Added Unreleased entries for dashboard, executor, and lock fixes.
