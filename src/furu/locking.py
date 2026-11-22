import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

_POLL_S = 0.01


@contextmanager
def short_lock(path: Path, *, timeout_s: float = 30.0) -> Generator[None]:
    """Hold ``path``, a file that exactly one process can create, for a few writes.

    Nothing breaks a lock whose holder was killed: past ``timeout_s`` this
    raises and names the file to delete.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"{path} was held for over {timeout_s:g} s, though it is held "
                    "only for a few writes, so its holder was likely killed. "
                    "Delete it if no furu process is running beside it."
                ) from None
            time.sleep(_POLL_S)
    try:
        yield
    finally:
        path.unlink()
