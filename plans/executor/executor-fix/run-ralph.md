# How to run the Ralph loop

From repo root:

```bash
ralph --file plans/executor-fix/entrypoint.md --max-iterations 200 --completion-promise COMPLETE --no-commit
```

Notes:
- Use `--no-commit` so the model performs `git commit` and `git push` explicitly (per entrypoint rules).
- You can monitor progress with:
  ```bash
  ralph --status
  ```
- You can inject guidance without stopping:
  ```bash
  ralph --add-context "Focus on Phase 1A then Phase 1C; avoid refactors."
  ```
