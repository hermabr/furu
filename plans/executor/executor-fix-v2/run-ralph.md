# Run the Ralph loop (no auto-commit)

From repo root:

```bash
ralph \
  --file plans/executor-fix-v2/entrypoint.md \
  --max-iterations 200 \
  --completion-promise COMPLETE \
  --no-commit \
  --allow-all
```

Notes:
- `--no-commit` is important because the model is instructed to commit/push itself.
- Use `ralph --status` in another terminal to monitor progress.
- Use `ralph --add-context "<hint>"` to steer if the loop stalls.
