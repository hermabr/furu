# Run the Ralph loop (no auto-commit)

From repo root:

```bash
ralph \
  --file plans/executor-fix-v5/entrypoint.md \
  --max-iterations 200 \
  --completion-promise COMPLETE \
  --no-commit \
  --allow-all
```

Tips:
- Use `ralph --status` to monitor.
- Use `ralph --add-context "<hint>"` if it stalls.
