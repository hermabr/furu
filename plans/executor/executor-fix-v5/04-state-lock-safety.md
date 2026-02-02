# 04 — State lock safety: never unlink a lock you don't own

## Bug (blocking per review)
`StateManager.update_state` calls `release_lock` in `finally`, and `release_lock` unlinks lock path even when `fd is None`.
If `_acquire_lock_blocking` times out, this can delete **another process’s lock** and allow concurrent writers.

## Desired behavior
- Only unlink the lock file if the current process actually owns it.
- If acquisition failed (fd is None), do not unlink.
- Ideally, lock files contain owner metadata (pid + random token) and unlink is conditional.

## Implementation steps
- [x] In `src/furu/storage/state.py`:
  - Ensure `release_lock` checks `fd is not None` before unlinking.
  - If you embed owner token in lock file, verify token matches before unlink.
- [x] Add regression test:
  - Simulate process A holds lock (create lock file)
  - process B calls update_state and times out
  - assert lock file still exists afterward

## Acceptance criteria
- No code path unlinks a lock without ownership.
- Test prevents regression.

## Progress log
- Status: [x] done
- Notes: Guarded release_lock on fd, handled lock read races, and added timeout regression test.
