I like ambitious ideas, simple systems, and software that feels obvious. Do not preserve complexity just because it already exists. Do not introduce machinery because it looks architecturally impressive. Understand the real constraint, then fight for the smallest model that makes the correct behavior unsurprising.

I love to build. I focus on building complex things as simple as possible. I love to find ways to reduce complexity when solving problems.

Channel "measure twice, cut once", "yagni" and "one-liner solutions". Fight scope creep. Try to honor the dev's intent in both a minimal and realistic fashion.

Furu is pre-release with no users. Prefer solutions that will improve the long term maintainability of the codebase. Never keep a feature just because it already exists. Break APIs freely and don't add deprecation shims.

Furu is an all-in-one experiment runner (DAG definition, execution, and provenance) built primarily for AI agents, with humans as secondary users. Favor strict guarantees over convenience—requiring clean, committed git state is cheap for an agent—and keep everything needed to reproduce an artifact in repo source rather than external manager state. Prefer declarative DAGs the manager can plan up front, while keeping dynamic discovery inside `create()` possible.

Think of these instructions less as "hard rules", more as "good defaults". If a rule here fights the task in front of you, say so loudly and get a human sign-off before breaking it.

We use `uv`, such as `uv run python` and you should always check that `uv run ruff check && uv run ty check && uv run pytest` pass.

Relative `furu-data/` paths anchor at the main git worktree root, so run experiments via `pytest` or explicit `FURU_DIRECTORIES__*` env vars—never ad-hoc scripts that pollute shared data.
