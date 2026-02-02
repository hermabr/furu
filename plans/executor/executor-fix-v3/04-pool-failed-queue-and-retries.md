# 04 — Pool failed queue handling + bounded retries

## Reported issues (blocking + contradiction)
- Controller can spin forever if tasks are in `queue/failed` and `_task_known` prevents requeue while ready remains non-empty.
- Some implementations abort on any failed queue entry unconditionally, while workers write failed entries for any non-success state.
  => Pool fails fast even when `retry_failed=True`, contradicting "retry failures in executors".

## Desired behavior
- Controller must actively inspect `queue/failed` each tick.
- For each failed task:
  - if failure_kind == "protocol": abort immediately with clear message
  - if failure_kind == "compute":
    - if retry_failed=True and attempt < max_compute_retries: requeue to todo
    - else abort with clear message
- Controller must not spin forever on failed entries.

## Implementation steps
- [x] Implement `_scan_failed_tasks(run_dir)` returning parsed failure records.
- [x] Implement `_handle_failed_tasks(...)` that:
  - requeues eligible compute failures when retries enabled
  - aborts for protocol failures or exhausted retries
- [x] Ensure `_task_known` doesn’t permanently block retriable failures:
  - a task in failed queue should be considered “known” but either retried or aborts promptly
- [x] Add strong error messages:
  - include task path, hash, spec_key, worker_id, error, attempt count

## Acceptance criteria
- No infinite spin when failed queue entries exist.
- Compute failures are retried when retry_failed=True (bounded by max_compute_retries).
- Protocol failures abort immediately.

## Tests
- [ ] Inject a failed queue entry (protocol) and ensure controller aborts immediately.
- [ ] Simulate a compute failure with retry_failed=True and ensure it is requeued and retried, then succeeds (or exhausts retries).

## Progress log
- Status: [ ] not started / [x] in progress / [ ] done
- Notes:
  - Added failed queue scanning, retry requeue path, and detailed failure reporting.
