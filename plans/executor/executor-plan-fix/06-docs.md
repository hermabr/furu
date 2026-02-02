# 06 — Docs: breaking changes + always_rerun semantics

## Why

User-visible changes should be documented:
- removal of `load_or_create`
- `get()` signature changed (any per-call `retry_failed` override removed)
- executor `always_rerun` semantics may differ (often “once per executor run” due to scheduler caching)

Relevant note:
- `src/furu/core/furu.py:335` mentioned as potential user-visible change (retry_failed override removal)

## Tasks

- [x] Add a docs/changelog entry describing breaking changes:
  - `load_or_create` removed
  - `get()` is the only API
  - retry_failed is config-based (no per-call override)
  - how executor strictness works

- [x] Document executor `always_rerun` behavior:
  - if “once per executor run” is intentional, say so and why
  - if not, open a follow-up plan item (do not silently change semantics without tests)

## Acceptance criteria

- A reader can understand behavioral differences without reading code.

## Progress log

| Date | Summary |
|---|---|
| 2026-01-22 | Document breaking API changes and executor always_rerun semantics in README/CHANGELOG. |

## Plan changes

| Date | Change | Why |
|---|---|---|
|  |  |  |
