I like ambitious ideas, simple systems, and software that feels obvious. Do not preserve complexity just because it already exists. Do not introduce machinery because it looks architecturally impressive. Understand the real constraint, then fight for the smallest model that makes the correct behavior unsurprising.

I love to build. I focus on building complex things as simple as possible. I love to find ways to reduce complexity when solving problems.

Channel "measure twice, cut once", "yagni" and "one-liner solutions". Fight scope creep. Try to honor the dev's intent in both a minimal and realistic fashion.

Furu is pre-release with no users. Prefer solutions that will improve the long term maintainability of the codebase. Never keep a feature just because it already exists. Break APIs freely and don't add deprecation shims.

Think of these instructions less as "hard rules", more as "good defaults". If a rule here fights the task in front of you, say so loudly and get a human sign-off before breaking it.

We use `uv`, such as `uv run python` and you should always check that `uv run ruff check && uv run ty check && uv run pytest` pass.

## Where to spend care

Some decisions are expensive to change later: spec identity and hashing, the on-disk layout and result manifest, migration semantics, and protocols between processes (ownership, leases, handoff). For these, write down the decisions and their trade-offs and get my sign-off before writing the code. Everything else (orchestration internals, logging, config plumbing, scaling) is cheap to change: make the call yourself, say what you chose in the PR description, and move on.

Before calling something done, do a subtraction pass: what can be deleted, inlined, or left out? Propose removing features nobody uses rather than polishing them.

## Code

- Inline a helper with one caller unless it names a real concept.
- Comments say why, never what. No docstrings that restate the signature.
- Prefer a required setting to an `Optional` one with a default, and add knobs only for a caller that needs them.
- Extend the existing mechanism instead of adding a parallel one, and fix a problem where it lives instead of globally.
- No ride-along features: anything the task didn't ask for goes in a separate PR or a note.
- Tests check behavior through the public API, not internal structure or message shapes, so a redesign doesn't mean rewriting the suite.

## Pull requests

Keep PRs under about 500 changed lines outside tests; split larger ones before asking for review. Open the PR description with the decisions I need to make, if any.
