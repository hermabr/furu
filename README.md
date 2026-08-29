# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Provenance

Every result records the git commit it was computed from. On submit, furu
commits your working tree — tracked edits **and every untracked file that is
not ignored** — in a scratch index, so your checkout never moves, pins it at
`refs/furu/<sha>` and pushes it to `origin`. Anything not in `.gitignore`
leaves the machine: a stray `.env` or credentials file becomes a leak, so
ignore it first. The first time a snapshot is pushed, furu logs the untracked
files it shipped.

Anyone with repo access can later fetch the exact code that ran:

```bash
git fetch origin refs/furu/<sha>
git diff <sha>~1 <sha>            # the uncommitted edits at submit time
git worktree add ../rerun <sha>   # the exact tree that ran
```

`<sha>` is `snapshot_id` in `provenance.json`; it equals `git.commit` when the
tree was clean. Refs under `refs/furu/` are not branches, so they never show up
in `git branch -r` or the GitHub branch dropdown.

Settings live under `[tool.furu.provenance]` in `pyproject.toml`, or as
`FURU_PROVENANCE__<NAME>` environment variables:

| Setting              | Default  | Meaning                                                                                  |
| -------------------- | -------- | ---------------------------------------------------------------------------------------- |
| `snapshot`           | `true`   | Build the snapshot commit. Off: workers run the live tree and `snapshot_id` is null.     |
| `push`               | `true`   | Push to `origin`. Off: the local ref is still created, but nothing leaves the machine.   |
| `max_snapshot_bytes` | `256MiB` | Refuse larger snapshots, listing the largest files, before anything is committed or pushed. |
