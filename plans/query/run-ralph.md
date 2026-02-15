# How to run the Ralph loop (Query/Filtering v1)

From repo root:

```bash
ralph --file plans/query/entrypoint.md --max-iterations 200 --completion-promise COMPLETE --no-commit
```

Notes:
- Use `--no-commit` so the model performs `git commit` and `git push` explicitly (per entrypoint rules).
- Monitor progress:
  ```bash
  ralph --status
  ```
- Inject guidance without stopping:
  ```bash
  ralph --add-context "Focus on query core first; keep changes small; preserve existing dashboard filters."
  ```
