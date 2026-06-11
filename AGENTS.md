# Furu

This is the repository for Furu, the best and most efficient way to run and configure reproducible experiments. The goal is a end-to-end library for humans and agents.

## Project Snapshot

Furu is a minimal and flexible plugin for running and configuring reproducible experiments.

This repository is a VERY EARLY WIP and does not have any users. Proposing sweeping changes that improve long-term maintainability is encouraged.

## Maintainability

Long term maintainability is a core priority. If you add new functionality, first check if there is shared logic that can be extracted to a separate module. Duplicate logic across multiple files is a code smell and should be avoided. Don't be afraid to change existing code. Don't take shortcuts by just adding local logic to solve a problem. Make an active effort to implement all changes as directed and minimally as possible.

## Task Completion Requirements

- All of `uv run ruff format`, `uv run ruff check`, `uv run ty check` and `uv run pytest`, should pass before considering tasks completed.
